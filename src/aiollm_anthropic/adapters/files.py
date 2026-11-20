"""Adapter of shared files to Anthropic's Files API."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from datetime import UTC
from typing import Final, Self

from pydantic import MISSING

from aiollm.adapters.base import BaseFilesAdapter
from aiollm.adapters.validation import checked_options, request_errors
from aiollm.types.requests.content import ProviderFile
from aiollm.types.requests.generation import NoProviderOptions
from aiollm.types.requests.upload import FileUpload
from aiollm.types.responses.files import StoredFile

from aiollm_anthropic.api import PROVIDER
from aiollm_anthropic.methods.files import (
    CreateFile,
    DeleteFile,
    DownloadFile,
    ListFiles,
    RetrieveFile,
)
from aiollm_anthropic.types.files import (
    FileMetadata,
    ListFilesResponse,
)

DEFAULT_PAGE_SIZE: Final = 100
"""Files asked for in one page of ``client.files.list()``."""


class AnthropicFilesAdapter(
    BaseFilesAdapter[FileMetadata, ListFilesResponse],
):
    """Stores files with Anthropic's ``/files``.

    ``expires_in_seconds`` is sent as it is. The list goes newest first,
    ``page_size`` files a request, by the opaque ``next_page``. Uploads
    take no provider options.

    Attributes:
        page_size: Files asked for in one page of the list, 1 to 1000.

    Example:
        >>> files = FilesResource(session, AnthropicFilesAdapter())
        >>> stored = await files.upload(Path("report.pdf"))
    """

    provider = PROVIDER

    def __init__(self, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        """Store the page size.

        Args:
            page_size: Files asked for in one page of the list.
        """
        self.page_size = page_size

    def with_options(self, options: NoProviderOptions, /) -> Self:
        """Check that no option is given; Anthropic's uploads take none.

        Args:
            options: The call's provider options, which must be empty.

        Returns:
            A copy of the adapter.

        Raises:
            RequestValidationError: A key is given.
        """
        checked_options(options, NoProviderOptions, self.provider)
        return copy.copy(self)

    def to_upload_method(self, request: FileUpload) -> CreateFile:
        """Build the method that stores a file.

        Args:
            request: The shared upload.

        Returns:
            ``CreateFile`` with the file and its expiry.

        Raises:
            RequestValidationError: Anthropic cannot take the upload,
                such as an expiry outside 3600 to 7 776 000 seconds.
        """
        with request_errors(self.provider):
            return CreateFile(
                file=request.file,
                expires_in_seconds=request.expires_in_seconds,
            )

    def to_retrieve_method(self, file_id: str) -> RetrieveFile:
        """Build the method that reads a file's metadata.

        Args:
            file_id: Anthropic's id of the file.

        Returns:
            ``RetrieveFile``.
        """
        return RetrieveFile(file_id=file_id)

    def to_list_method(self, cursor: str | None) -> ListFiles:
        """Build the method that reads one page of the stored files.

        Args:
            cursor: ``next_page`` of the page before, or ``None`` for
                the first page.

        Returns:
            ``ListFiles`` of ``page_size`` files.
        """
        return ListFiles(
            limit=self.page_size,
            page=MISSING if cursor is None else cursor,
        )

    def to_page(
        self, raw: ListFilesResponse
    ) -> tuple[Sequence[FileMetadata], str | None]:
        """Take the files of a page and where the next one starts.

        Args:
            raw: The page.

        Returns:
            Its files and its ``next_page``.
        """
        return raw.data, raw.next_page

    def to_delete_method(self, file_id: str) -> DeleteFile:
        """Build the method that deletes a file.

        Args:
            file_id: Anthropic's id of the file.

        Returns:
            ``DeleteFile``.
        """
        return DeleteFile(file_id=file_id)

    def to_download_method(self, file_id: str) -> DownloadFile:
        """Build the method that streams a file's content.

        Args:
            file_id: Anthropic's id of the file.

        Returns:
            ``DownloadFile``; Anthropic refuses a file it did not make
            (``downloadable`` false).
        """
        return DownloadFile(file_id=file_id)

    def to_stored_file(self, raw: FileMetadata) -> StoredFile[FileMetadata]:
        """Convert Anthropic's file metadata into the shared metadata.

        Args:
            raw: Anthropic's metadata.

        Returns:
            Its id, name, ``size_bytes`` as ``size`` and its times in
            UTC.
        """
        expires_at = raw.expires_at
        return StoredFile[FileMetadata](
            ref=ProviderFile(provider=self.provider, id=raw.id),
            filename=raw.filename,
            size=raw.size_bytes,
            created_at=raw.created_at.astimezone(UTC),
            expires_at=None
            if expires_at is None
            else expires_at.astimezone(UTC),
            raw=raw,
        )
