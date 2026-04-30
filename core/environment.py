"""TauBenchEnv — τ²-bench (Python package ``tau2``, tau3 line) wrapper for airline or retail.

Domain JSON and policies load from ``$TAU2_DATA_DIR/tau2/domains/<domain>/`` or from the
tau2 source tree when installed editable. See tau2-bench README for ``TAU2_DATA_DIR``.
"""
from __future__ import annotations

from typing import Literal

from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType

Domain = Literal["airline", "retail"]

# Hardcoded write-tool sets are kept only as a drift tripwire: at startup we
# assert they equal the set tau2 derives from ``ToolType.WRITE`` decorators.
# The runtime ``is_write_tool()`` reads the live derivation, so a future tau2
# bump that adds/renames a write tool fails loudly here instead of silently
# misclassifying it.
AIRLINE_WRITE_TOOLS = frozenset({
    "book_reservation", "cancel_reservation",
    "update_reservation_flights", "update_reservation_passengers",
    "update_reservation_baggages", "send_certificate",
})

RETAIL_WRITE_TOOLS = frozenset({
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
})

AIRLINE_VERIFICATION_TOOLS = frozenset({"get_user_details", "get_reservation_details"})
RETAIL_VERIFICATION_TOOLS = frozenset({"get_user_details", "get_order_details"})


class TauBenchEnv:
    def __init__(self, domain: Domain = "airline"):
        self.domain: Domain = domain
        if domain == "airline":
            from tau2.domains.airline.data_model import FlightDB
            from tau2.domains.airline.environment import get_environment
            from tau2.domains.airline.utils import AIRLINE_DB_PATH

            self._build_env = get_environment
            self._db_path = AIRLINE_DB_PATH
            self._db_cls = FlightDB
            self._verification_tools = AIRLINE_VERIFICATION_TOOLS
            self._expected_write_tools = AIRLINE_WRITE_TOOLS
        elif domain == "retail":
            from tau2.domains.retail.data_model import RetailDB
            from tau2.domains.retail.environment import get_environment
            from tau2.domains.retail.utils import RETAIL_DB_PATH

            self._build_env = get_environment
            self._db_path = RETAIL_DB_PATH
            self._db_cls = RetailDB
            self._verification_tools = RETAIL_VERIFICATION_TOOLS
            self._expected_write_tools = RETAIL_WRITE_TOOLS
        else:
            raise ValueError(f"unknown domain: {domain!r}, expected 'airline' or 'retail'")

        self._env: Environment | None = None
        self.reset()
        assert self._env is not None

        # Schemas and policy text depend on the toolkit class and policy.md,
        # not DB state — capture them once. Tool.openai_schema rebuilds the
        # Pydantic JSON schema on each access (~27ms × every agent turn),
        # so caching here saves ~30 minutes across a full sweep.
        self._wiki_cache: str = self._env.policy
        self._tools_info_cache: list[dict] = [t.openai_schema for t in self._env.get_tools()]

        derived_writes = frozenset(
            name for name in self._env.tools.tools
            if self._env.tools.tool_type(name) == ToolType.WRITE
        )
        assert derived_writes == self._expected_write_tools, (
            f"tau2 write-tool drift on {domain}: derived={sorted(derived_writes)} "
            f"vs expected={sorted(self._expected_write_tools)}"
        )

    @property
    def verification_tools(self) -> frozenset[str]:
        return self._verification_tools

    def reset(self) -> None:
        """Load a fresh DB and environment. Call once per attack bundle (not per session)."""
        fresh_db = self._db_cls.load(self._db_path)
        self._env = self._build_env(db=fresh_db)

    def _ensure(self) -> Environment:
        assert self._env is not None
        return self._env

    @property
    def wiki(self) -> str:
        return self._wiki_cache

    @property
    def tools_info(self) -> list[dict]:
        return self._tools_info_cache

    @property
    def data(self):
        """Underlying Pydantic DB (mutated by tools). Prefer ``exec_tool`` for agent calls."""
        return self._ensure().tools.db

    def exec_tool(self, name: str, kwargs: dict) -> str:
        env = self._ensure()
        if not env.tools.has_tool(name):
            return f"Error: unknown tool {name}"
        try:
            resp = env.use_tool(name, **kwargs)
        except Exception as e:
            return f"Error: {e}"
        return Environment.to_json_str(resp)

    def is_write_tool(self, name: str) -> bool:
        if not self._ensure().tools.has_tool(name):
            return False
        return self._ensure().tools.tool_type(name) == ToolType.WRITE
