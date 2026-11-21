"""Users, roles and the one access rule every layer shares.

Access rule (checked in the vector filter, the keyword index and again
after fusion, so a bug in one layer cannot leak a document):

    user may read doc  <=>  doc.company in user.companies (or doc is group-wide)
                       and  doc.roles  intersects  user.roles
"""

import re
from dataclasses import dataclass
from enum import Enum

from . import db
from .config import GROUP_COMPANY


@dataclass(frozen=True)
class User:
    """A signed-in person. Companies and roles come from the identity provider."""

    user_id: str
    name: str
    companies: frozenset[str]
    roles: frozenset[str]


@dataclass(frozen=True)
class DocAccess:
    """Access tags read from a document's front matter."""

    company: str
    roles: frozenset[str]


# Sample directory: the seed for Postgres, and the directory itself when
# Postgres is off. In production this comes from Entra ID / Microsoft Graph groups.
_USERS = {
    "sara": User("sara", "Sara (HR, Alpha)", frozenset({"alpha-construction"}), frozenset({"employee", "hr"})),
    "omar": User("omar", "Omar (Procurement, Alpha)", frozenset({"alpha-construction"}), frozenset({"employee", "procurement"})),
    "lina": User("lina", "Lina (Finance, Beta)", frozenset({"beta-properties"}), frozenset({"employee", "finance"})),
    "faisal": User("faisal", "Faisal (Finance, Alpha)", frozenset({"alpha-construction"}), frozenset({"employee", "finance"})),
    "khalid": User("khalid", "Khalid (Site engineer, Alpha)", frozenset({"alpha-construction"}), frozenset({"employee", "site_engineer"})),
    "noura": User("noura", "Noura (Executive, Group)", frozenset({"alpha-construction", "beta-properties"}), frozenset({"employee", "executive"})),
}


def get_user(user_id: str) -> User:
    """Return a known user or raise KeyError. With Postgres, reads current roles on every call."""
    if db.enabled():
        row = db.fetch_user(user_id)
        if row is None:
            raise KeyError(f"unknown user '{user_id}'")
        return _user_from_row(row)
    if user_id not in _USERS:
        raise KeyError(f"unknown user '{user_id}'; known: {', '.join(sorted(_USERS))}")
    return _USERS[user_id]


def list_users() -> list[User]:
    """All users, sorted by id."""
    if db.enabled():
        return [_user_from_row(row) for row in db.fetch_users()]
    return [_USERS[key] for key in sorted(_USERS)]


def _user_from_row(row: db.UserRow) -> User:
    return User(row.user_id, row.name, frozenset(row.companies), frozenset(row.roles))


def allowed_companies(user: User) -> list[str]:
    """Companies whose documents the user may see, including group-wide ones."""
    return sorted(user.companies | {GROUP_COMPANY})


# Company names as people write them: (full name, short name); the full name is
# the label. A stand-in for the directory; production reads them from the HR system.
_COMPANY_NAMES = {
    "alpha-construction": ("Alpha Construction", "Alpha"),
    "beta-properties": ("Beta Properties", "Beta"),
}
_GROUP_LABEL = "All group companies"

# Full names match in any letter case. Short names are ordinary words too
# ("a beta purchase", "alpha-numeric"), so they match only capitalised and not
# before a hyphen.
_COMPANY_RE = {
    company: re.compile(rf"(?i:\b{re.escape(full)}\b)|\b{re.escape(short)}\b(?!-)")
    for company, (full, short) in _COMPANY_NAMES.items()
}


def companies_named(text: str) -> set[str]:
    """Company ids a text mentions by name: 'at Alpha Construction' -> {'alpha-construction'}."""
    return {company for company, pattern in _COMPANY_RE.items() if pattern.search(text)}


def without_company_names(text: str) -> str:
    """The text with company names removed, so a name scopes a search but does not score it."""
    for pattern in _COMPANY_RE.values():
        text = pattern.sub(" ", text)
    return text


def company_label(company: str) -> str:
    """How a company is shown to people and to the model."""
    if company == GROUP_COMPANY:
        return _GROUP_LABEL
    return _COMPANY_NAMES.get(company, (company,))[0]


def can_read(user: User, access: DocAccess) -> bool:
    """The single access rule. See the module docstring."""
    if access.company not in allowed_companies(user):
        return False
    return bool(access.roles & user.roles)


# ---- Postgres as the source of truth -----------------------------------------
#
# Front matter seeds a document's access once. After that the documents table
# decides: revoking access there takes effect on the next question, and a
# document with no row is unreadable. Widening access in Postgres shows up only
# after a re-index, because the vector payload and keyword filter still carry the
# old tags; that direction fails closed.


def current_access(stamped: dict[str, DocAccess]) -> dict[str, DocAccess]:
    """Access to enforce now for these doc ids. Postgres off: the stamped tags."""
    if not db.enabled():
        return stamped
    rows = db.fetch_documents(sorted(stamped))
    return {row.doc_id: DocAccess(row.company, frozenset(row.roles)) for row in rows}


def seed_directory(overwrite: db.Overwrite, documents: dict[str, tuple[str, DocAccess]]) -> None:
    """Create tables and seed users and document access.

    documents maps doc id to (title, access from front matter).
    KEEP seeds users only into an empty table, so a deleted user stays deleted,
    and adds missing documents. RESTORE resets every seed user and document.
    """
    db.create_schema()
    if overwrite is db.Overwrite.RESTORE or not db.fetch_users():
        rows = [db.UserRow(u.user_id, u.name, sorted(u.companies), sorted(u.roles)) for u in _USERS.values()]
        db.insert_users(rows, overwrite)
    register_documents(overwrite, documents)


def register_documents(overwrite: db.Overwrite, documents: dict[str, tuple[str, DocAccess]]) -> None:
    """Store access rows for documents (see seed_directory)."""
    rows = [
        db.DocumentRow(doc_id, title, access.company, sorted(access.roles))
        for doc_id, (title, access) in sorted(documents.items())
    ]
    db.insert_documents(rows, overwrite)


class RoleChange(str, Enum):
    """Grant or revoke."""

    GRANT = "grant"
    REVOKE = "revoke"


def set_role(user_id: str, role: str, change: RoleChange) -> None:
    """Grant or revoke one role in Postgres. Raises KeyError for an unknown user."""
    changed = db.add_user_role(user_id, role) if change is RoleChange.GRANT else db.remove_user_role(user_id, role)
    if not changed:
        raise KeyError(f"unknown user '{user_id}'")

