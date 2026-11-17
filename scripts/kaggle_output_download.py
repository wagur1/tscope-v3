"""Download Kaggle kernel outputs into an empty directory with path-traversal guards and UTF-8 logs."""
import time
from pathlib import Path


def download_kernel_outputs(api, kernel_id: str, destination: Path, max_retries: int = 3) -> None:
    import requests
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest

    destination = Path(destination).resolve()
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("Kernel download requires a new empty directory")

    owner, slug, _ = api.parse_kernel_string(kernel_id)
    page_token = None
    seen_tokens = set()

    while True:
        with api.build_kaggle_client() as client:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            api._set_paging(request, 20, page_token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)

        for item in response.files or []:
            raw_name = str(item.file_name).replace("\\", "/")
            relative = Path(raw_name)
            target = destination / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or ":" in raw_name
                or not target.resolve().is_relative_to(destination)
            ):
                raise ValueError("Unsafe kernel output filename")

            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_target = target.with_name(f".{target.name}.part")
            last_err = None
            for attempt in range(max(1, max_retries)):
                try:
                    if tmp_target.exists():
                        tmp_target.unlink()
                    with requests.get(item.url, stream=True, timeout=(30, 180)) as result:
                        if result.status_code >= 400:
                            raise RuntimeError(f"HTTP {result.status_code} downloading {relative.as_posix()}")
                        with tmp_target.open("xb") as handle:
                            for chunk in result.iter_content(1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    if target.exists():
                        raise FileExistsError(f"Duplicate output file {relative.as_posix()}")
                    tmp_target.replace(target)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    if tmp_target.exists():
                        tmp_target.unlink(missing_ok=True)
                    if attempt + 1 < max_retries:
                        time.sleep(2.0 * (attempt + 1))
            if last_err is not None:
                raise RuntimeError(f"Failed to download {relative.as_posix()}: {type(last_err).__name__}") from None

        if response.log:
            with (destination / f"{slug}.log").open("a", encoding="utf-8", newline="") as handle:
                handle.write(response.log)

        page_token = response.next_page_token
        if not page_token:
            return
        if page_token in seen_tokens:
            raise ValueError("Repeated output pagination token")
        seen_tokens.add(page_token)
