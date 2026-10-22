import numpy as np

from lanerl_jax.modern.bc.replay import _cooldown_events


class Fields:
    def __init__(self, qualified=True, proof_event=8, duplicate=False):
        proof = dict(timer_semantics_validated=qualified,
                     slot_array_index_semantics_validated=qualified, slot_lookup_operand=0)
        self.payloads = {"cooldown.received_operands": [proof],
                         "cooldown.cast_start": [dict(available=True, clock_alignment_qualified=False)]}
        self.native = {"fields": {}}
        values = {"cooldown.slot": 0, "cooldown.remaining_at_receipt": 8.,
                  "cooldown.received_duration": 8., "cooldown.mode": 0}
        for name in (*values, *self.payloads):
            n = 2 if duplicate and name == "cooldown.slot" else 1
            table = dict(event_index=np.full(n, proof_event if name == "cooldown.received_operands" else 8),
                         time_seconds=np.full(n, 10.))
            if name in values:
                table["values"] = np.full(n, values[name])
            self.native["fields"][name] = table

    def rows(self, name, family=None):
        table = self.native["fields"].get(name)
        if table is None:
            return [], [], []
        n = len(table["time_seconds"])
        return np.arange(n), table["time_seconds"], np.full(n, 99)

    def value(self, name, row):
        return self.payloads[name][row]


def test_operand_names_do_not_qualify_timer_semantics():
    events, audit = _cooldown_events(Fields(qualified=False), 99)
    assert events == [] and audit["rejected_broadcast_packets"] == 1
    assert audit["cast_schedule_records"] == 1 and audit["clock_qualified_cast_schedules"] == 0
    assert audit["cast_schedules_consumed"] is False


def test_qualified_packet_is_joined_by_source_identity():
    events, audit = _cooldown_events(Fields(), 99)
    assert events == [(10., 0, 8., 8., 0)]
    assert audit["qualified_broadcast_packets"] == 1
    events, _ = _cooldown_events(Fields(proof_event=9), 99)
    assert events == []  # identical receipt time does not associate different packets


def test_missing_proof_duplicate_rows_and_wrong_actor_are_not_accepted():
    f = Fields()
    del f.native["fields"]["cooldown.received_operands"]
    assert _cooldown_events(f, 99)[0] == []
    assert _cooldown_events(Fields(duplicate=True), 99)[0] == []
    assert _cooldown_events(Fields(), 100)[0] == []


def test_slot_disagreement_or_nonfinite_values_are_not_accepted():
    f = Fields()
    f.payloads["cooldown.received_operands"][0]["slot_lookup_operand"] = 1
    assert _cooldown_events(f, 99)[0] == []
    f = Fields()
    f.native["fields"]["cooldown.remaining_at_receipt"]["values"][0] = np.nan
    assert _cooldown_events(f, 99)[0] == []
