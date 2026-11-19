"""Methods of Anthropic's Files API."""

from __future__ import annotations

from aiollm.enums import BodyFormat, ResponseFormat
from aiollm.methods.base import BaseAPIMethod, BaseStreamMethod
from aiollm.providers.anthropic.methods.messages import VERSION_HEADERS
from aiollm.providers.anthropic.methods.params.files import (
    CreateFileParams,
    DeleteFileParams,
    DownloadFileParams,
    ListFilesParams,
    RetrieveFileParams,
)
from aiollm.providers.anthropic.types.files import (
    DeletedFile,
    FileMetadata,
    ListFilesResponse,
)


class CreateFile(CreateFileParams, BaseAPIMethod[FileMetadata]):
    """Upload a file, to refer to it by its ``id`` in later requests.

    The upload is never retried: ``POST /files`` is not idempotent, and a
    retry after a lost answer would store the file twice. A timeout or a
    lost connection may therefore leave a file at Anthropic whose ``id``
    you never got; give the file ``expires_in_seconds`` where that
    matters.

    Every key of the workspace sees its files: treat a file ``id`` as a
    server-side reference and never take one from an end user.

    Example:
        >>> file = await client.send_request(
        ...     CreateFile(file=Path("report.pdf"), expires_in_seconds=3600)
        ... )
        >>> file.id
    """

    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __request_headers__ = VERSION_HEADERS
    __retryable__ = False


class ListFiles(ListFilesParams, BaseAPIMethod[ListFilesResponse]):
    """List the files of the workspace, newest first, a page at a time.

    Pass the page's ``next_page`` as ``page`` while it is not ``None``.
    ``ids`` (up to 100) asks for those files only, in one page, and
    cannot be combined with ``page`` or ``limit``. The id of a deleted
    or unknown file is left out, while one that is not a valid file id
    fails the whole request with ``BadRequestError``.

    Example:
        >>> page = await client.send_request(ListFiles(limit=100))
        >>> while page.next_page is not None:
        ...     page = await client.send_request(
        ...         ListFiles(limit=100, page=page.next_page)
        ...     )
    """

    __request_path__ = "/files"
    __request_method__ = "GET"
    __request_query__ = frozenset({"ids", "limit", "page"})
    __request_headers__ = VERSION_HEADERS
    __retryable__ = True


class RetrieveFile(RetrieveFileParams, BaseAPIMethod[FileMetadata]):
    """Get the metadata of a file.

    Example:
        >>> file = await client.send_request(RetrieveFile(file_id="file_1"))
    """

    __request_path__ = "/files/{file_id}"
    __request_method__ = "GET"
    __request_headers__ = VERSION_HEADERS
    __retryable__ = True


class DeleteFile(DeleteFileParams, BaseAPIMethod[DeletedFile]):
    """Delete a file.

    It is retried on a transient failure, as the official SDKs do; a
    retry after a deletion whose answer was lost raises
    ``NotFoundError``, since the file is already gone.

    Example:
        >>> await client.send_request(DeleteFile(file_id="file_1"))
    """

    __request_path__ = "/files/{file_id}"
    __request_method__ = "DELETE"
    __request_headers__ = VERSION_HEADERS
    __retryable__ = True


class DownloadFile(DownloadFileParams, BaseStreamMethod[bytes]):
    """Stream the content of a file that a tool created.

    Only files with ``downloadable`` set (made by code execution or a
    skill) can be downloaded; any other raises ``BadRequestError``
    (``file_not_downloadable``), and an expired one ``NotFoundError``.
    Only opening the download is retried. ``Stream.save(path)`` writes it
    to a file.

    Example:
        >>> method = DownloadFile(file_id="file_1")
        >>> async with client.stream(method) as content:
        ...     await content.save(Path("chart.png"))
    """

    __request_path__ = "/files/{file_id}/content"
    __request_method__ = "GET"
    __request_headers__ = VERSION_HEADERS
    __response_format__ = ResponseFormat.BYTES
    __retryable__ = True
