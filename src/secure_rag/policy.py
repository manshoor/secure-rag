"""Purchase approval rules from the procurement policy.

Source: data/docs/alpha-procurement-policy.md, section "Purchase thresholds":

    below AED 10,000            one written quotation
    AED 10,000 to 100,000       three quotations, finance approves
    above AED 100,000           open tender, an executive approves

The policy names no approver below AED 10,000; this code asks finance (or an
executive) there too, so every purchase has a second human check.
"""

from decimal import ROUND_HALF_UP, Decimal

# Thresholds, in AED, from the policy section above.
SINGLE_QUOTE_BELOW_AED = Decimal("10000")
EXECUTIVE_APPROVAL_AED = Decimal("100000")

# Money is rounded per line to fils (0.01 AED), half up; totals add rounded lines.
MONEY_QUANT = Decimal("0.01")
MONEY_ROUNDING = ROUND_HALF_UP

_FINANCE_ROLES = frozenset({"finance", "executive"})
_EXECUTIVE_ROLES = frozenset({"executive"})


def budget_approver_roles(total: Decimal) -> frozenset[str]:
    """Who may give the finance approval: an executive above the threshold, else finance."""
    if total > EXECUTIVE_APPROVAL_AED:
        return _EXECUTIVE_ROLES
    return _FINANCE_ROLES


def quotes_required(total: Decimal) -> str:
    """The quotation rule for this amount, as the policy states it."""
    if total < SINGLE_QUOTE_BELOW_AED:
        return "one written quotation"
    if total <= EXECUTIVE_APPROVAL_AED:
        return "three written quotations"
    return "open tender"


def money(value: Decimal) -> Decimal:
    """Round to fils with the policy rounding rule."""
    return value.quantize(MONEY_QUANT, rounding=MONEY_ROUNDING)
