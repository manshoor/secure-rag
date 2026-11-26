"""Specialist agents: each recommends one decision, code checks it, rules decide when it cannot.

Handlers read and compute only. Nothing here writes state, outbox, budget or audit:
phase 3 applies an outcome in the workflow.
"""
