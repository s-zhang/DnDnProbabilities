import pmf
from dnd import die, d


def test_addition():
	test_sum = die(1) + die(4)
	assert 0.25 == test_sum.p(3)
	assert 0.25 == test_sum.p(5)

def test_advantage():
	assert 0.28 == d(5).adv().p(4)

def test_greater_or_equal_than():
	assert 0.08000000000000002 == d(5).ge(3).p(2)
	assert 0.27999999999999997 == d(5).ge(3).p(3)

def test_joint_pmf():
	test_joint = pmf.joint([d(2), d(4)])
	assert 0.125 == test_joint.p((1, 2))

def test_equal():
	assert 0.25 == (d(4) == d(4)).p(True)

def test_union_generic():
	test_union = pmf.table({"a": 0.1, "b": 0.2}).union(pmf.table({"b": 0.4, "c": 0.3}))
	assert 0.6000000000000001 == test_union.p("b")
	assert 0.1 == test_union.p("a")

def test_union_int():
	test_union = d(2).union(d(2) + d(1))
	assert 1.0 == test_union.p(2)

def test_scale_probability():
	test_scale = d(5).scale_probability(0.5)
	assert 0.1 == test_scale.p(2)

def test_if():
	test_if = pmf.if_(d(5) == 1, d(1), d(1) + d(1))
	assert 0.2 == test_if.p(1)
	assert 0.8 == test_if.p(2)

def test_radd():
	assert (1 + d(3)).p(3) == (d(3) + 1).p(3)
