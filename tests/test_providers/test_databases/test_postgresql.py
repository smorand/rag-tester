"""Tests for PostgreSQLProvider using mocked psycopg connections."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
)
from rag_tester.providers.databases.postgresql import PostgreSQLProvider

VALID_CONN = "postgresql://user:pass@localhost:5432/mydb/test_table"


class _FakeCursor:
    """Async-context-manager cursor with queueable fetch results."""

    def __init__(self) -> None:
        self.fetchone_results: list[Any] = []
        self.fetchall_results: list[Any] = []
        self.rowcount = 0
        self.execute_calls: list[Any] = []
        self.execute_side_effect: Exception | None = None

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def execute(self, query: Any, *args: Any, **kwargs: Any) -> None:
        self.execute_calls.append((query, args, kwargs))
        if self.execute_side_effect is not None:
            raise self.execute_side_effect

    async def fetchone(self) -> Any:
        return self.fetchone_results.pop(0) if self.fetchone_results else None

    async def fetchall(self) -> Any:
        return self.fetchall_results.pop(0) if self.fetchall_results else []


class _FakeConn:
    """Async connection mock that yields _FakeCursor from cursor()."""

    def __init__(self) -> None:
        self.closed = False
        self.cursors: list[_FakeCursor] = []
        self.next_cursor: _FakeCursor | None = None
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.close = AsyncMock(side_effect=self._on_close)

    def _on_close(self) -> None:
        self.closed = True

    def cursor(self) -> _FakeCursor:
        cur = self.next_cursor or _FakeCursor()
        self.cursors.append(cur)
        # Advance: every call gets a fresh cursor unless a specific one is queued
        self.next_cursor = None
        return cur


class TestPostgreSQLParsing:
    def test_valid_connection_string(self) -> None:
        p = PostgreSQLProvider(VALID_CONN)
        assert p._user == "user"
        assert p._host == "localhost"
        assert p._port == 5432
        assert p._dbname == "mydb"
        assert p._table_name == "test_table"

    def test_missing_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="must start with 'postgresql://'"):
            PostgreSQLProvider("mysql://user:pass@localhost/db/tbl")

    def test_invalid_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be postgresql"):
            PostgreSQLProvider("postgresql://localhost/db/tbl")

    @pytest.mark.parametrize(
        "table",
        ["bad name", "bad-name", "bad;DROP", "bad'OR'1", "bad.dot"],
    )
    def test_hostile_table_name_rejected(self, table: str) -> None:
        with pytest.raises(ValueError, match="alphanumeric"):
            PostgreSQLProvider(f"postgresql://u:p@h:1/db/{table}")


class TestPostgreSQLOperations:
    @pytest.fixture
    def provider(self) -> PostgreSQLProvider:
        return PostgreSQLProvider(VALID_CONN)

    @pytest.fixture
    def fake_conn(self, provider: PostgreSQLProvider) -> _FakeConn:
        conn = _FakeConn()
        provider._conn = conn  # type: ignore[assignment]
        return conn

    async def test_get_connection_creates_once(self, provider: PostgreSQLProvider) -> None:
        fake = _FakeConn()
        with patch("psycopg.AsyncConnection.connect", new=AsyncMock(return_value=fake)) as connect:
            await provider._get_connection()
            await provider._get_connection()
            connect.assert_called_once()

    async def test_get_connection_failure_raises(self, provider: PostgreSQLProvider) -> None:
        with (
            patch("psycopg.AsyncConnection.connect", new=AsyncMock(side_effect=Exception("boom"))),
            pytest.raises(ConnectionError, match="Failed to connect"),
        ):
            await provider._get_connection()

    async def test_collection_exists_true(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchone_results = [{"exists": True}]
        fake_conn.next_cursor = cur
        result = await provider.collection_exists("test_table")
        assert result is True

    async def test_collection_exists_false(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchone_results = [{"exists": False}]
        fake_conn.next_cursor = cur
        result = await provider.collection_exists("test_table")
        assert result is False

    async def test_collection_exists_swallows_errors(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.execute_side_effect = RuntimeError("no table")
        fake_conn.next_cursor = cur
        assert await provider.collection_exists("test_table") is False

    async def test_create_collection_already_exists(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchone_results = [{"exists": True}]
        fake_conn.next_cursor = cur
        await provider.create_collection("test_table", dimension=4)

    async def test_create_collection_table_name_mismatch(self, provider: PostgreSQLProvider) -> None:
        with pytest.raises(DatabaseError, match="Table name mismatch"):
            await provider.create_collection("other", dimension=4)

    async def test_insert_empty_records_noop(self, provider: PostgreSQLProvider) -> None:
        await provider.insert("test_table", [])

    async def test_insert_table_name_mismatch(self, provider: PostgreSQLProvider) -> None:
        with pytest.raises(DatabaseError, match="Table name mismatch"):
            await provider.insert(
                "other",
                [{"id": "x", "text": "x", "embedding": [1.0]}],
            )

    async def test_query_returns_rows(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchall_results = [
            [
                {"id": "doc1", "text": "first", "metadata": None, "score": 0.9},
                {"id": "doc2", "text": "second", "metadata": None, "score": 0.5},
            ]
        ]
        fake_conn.next_cursor = cur
        results = await provider.query("test_table", [1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "doc1"

    async def test_delete_collection(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        fake_conn.next_cursor = cur
        await provider.delete_collection("test_table")
        assert len(cur.execute_calls) >= 1
        fake_conn.commit.assert_awaited()

    async def test_delete_all_returns_count(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchone_results = [{"count": 7}]
        fake_conn.next_cursor = cur
        deleted = await provider.delete_all("test_table")
        assert deleted == 7

    async def test_delete_all_empty_table(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchone_results = [{"count": 0}]
        fake_conn.next_cursor = cur
        deleted = await provider.delete_all("test_table")
        assert deleted == 0

    async def test_delete_by_ids_empty(self, provider: PostgreSQLProvider) -> None:
        assert await provider.delete_by_ids("test_table", []) == 0

    async def test_delete_by_ids_returns_rowcount(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.rowcount = 3
        fake_conn.next_cursor = cur
        deleted = await provider.delete_by_ids("test_table", ["a", "b", "c"])
        assert deleted == 3

    async def test_close(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        await provider.close()
        fake_conn.close.assert_awaited()

    async def test_close_when_already_none(self, provider: PostgreSQLProvider) -> None:
        provider._conn = None
        await provider.close()  # should not raise

    async def test_get_collection_info(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchone_results = [
            {"dimension": 4},  # dim query
            {"count": 5},  # count query
            {"comment": '{"foo": "bar"}'},  # comment query
        ]
        fake_conn.next_cursor = cur
        info = await provider.get_collection_info("test_table")
        assert info["name"] == "test_table"
        assert info["dimension"] == 4
        assert info["count"] == 5
        assert info["metadata"] == {"foo": "bar"}

    async def test_insert_success(self, provider: PostgreSQLProvider) -> None:
        # Drive a sequence of cursors: collection_exists -> True, dimension query -> 4, then insert
        fake_conn = _FakeConn()
        provider._conn = fake_conn  # type: ignore[assignment]

        # Multiple cursor() calls happen, each yields a fresh _FakeCursor.
        # We pre-queue cursors with the right responses.
        cursors = [
            _FakeCursor(),  # exists check
            _FakeCursor(),  # dimension query
            _FakeCursor(),  # actual INSERT execute
        ]
        cursors[0].fetchone_results = [{"exists": True}]
        cursors[1].fetchone_results = [{"dimension": 4}]
        idx = {"v": 0}

        def cursor_factory():
            cur = cursors[idx["v"]] if idx["v"] < len(cursors) else _FakeCursor()
            idx["v"] += 1
            fake_conn.cursors.append(cur)
            return cur

        fake_conn.cursor = cursor_factory  # type: ignore[assignment]

        await provider.insert(
            "test_table",
            [{"id": "doc1", "text": "hello", "embedding": [1.0, 0.0, 0.0, 0.0]}],
        )
        # Insert cursor must have been used (3rd one)
        assert len(cursors[2].execute_calls) >= 1

    async def test_query_with_filter_metadata(self, provider: PostgreSQLProvider, fake_conn: _FakeConn) -> None:
        cur = _FakeCursor()
        cur.fetchall_results = [[]]
        fake_conn.next_cursor = cur
        await provider.query(
            "test_table",
            [1.0, 0.0],
            top_k=5,
            filter_metadata={"src": "kb"},
        )
        assert len(cur.execute_calls) == 1
