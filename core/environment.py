"""TauBenchEnv — τ-bench wrapper (airline or retail) with persistent DB per attack bundle."""
from __future__ import annotations

from typing import Literal

Domain = Literal["airline", "retail"]

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
            from tau_bench.envs.airline.data import load_data
            from tau_bench.envs.airline.tools import ALL_TOOLS
            from tau_bench.envs.airline.wiki import WIKI

            self._load_data = load_data
            self.wiki = WIKI
            tools = ALL_TOOLS
            self._write_tools = AIRLINE_WRITE_TOOLS
            self._verification_tools = AIRLINE_VERIFICATION_TOOLS
        elif domain == "retail":
            from tau_bench.envs.retail.data import load_data
            from tau_bench.envs.retail.tools import ALL_TOOLS
            from tau_bench.envs.retail.wiki import WIKI

            self._load_data = load_data
            self.wiki = WIKI
            tools = ALL_TOOLS
            self._write_tools = RETAIL_WRITE_TOOLS
            self._verification_tools = RETAIL_VERIFICATION_TOOLS
        else:
            raise ValueError(f"unknown domain: {domain!r}, expected 'airline' or 'retail'")

        self._tools_map = {t.get_info()["function"]["name"]: t for t in tools}
        self.tools_info = [t.get_info() for t in tools]
        self._data = None

    @property
    def verification_tools(self) -> frozenset[str]:
        return self._verification_tools

    def reset(self):
        """Load fresh DB state. Called once per attack bundle, NOT per session."""
        self._data = self._load_data()

    @property
    def data(self):
        if self._data is None:
            self.reset()
        return self._data

    def exec_tool(self, name: str, kwargs: dict) -> str:
        if name not in self._tools_map:
            return f"Error: unknown tool {name}"
        try:
            return self._tools_map[name].invoke(data=self.data, **kwargs)
        except Exception as e:
            return f"Error: {e}"

    def is_write_tool(self, name: str) -> bool:
        return name in self._write_tools
