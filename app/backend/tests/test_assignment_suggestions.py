import pytest
from assignment_suggestions import AssignmentSegment, assignments_from_segments


def segment(top, start, end):
    return AssignmentSegment(top, str(top), start, end, .8, False, 'llm', 'Belegt')


def test_legacy_scalar_projection_does_not_pick_last_joint_top():
    assert assignments_from_segments(4, [segment(0, 0, 1), segment(1, 1, 2)]) == [0, None, 1, None]


def test_revisits_keep_identity_and_gaps():
    assert assignments_from_segments(4, [segment(1, 0, 0), segment(0, 2, 2), segment(1, 3, 3)]) == [1, None, 0, 1]


def test_invalid_range_not_clipped():
    with pytest.raises(ValueError):
        assignments_from_segments(2, [segment(0, -1, 1)])
