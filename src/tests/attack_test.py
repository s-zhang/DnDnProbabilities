from dnd import AttackBuilder, HitOutcome, TurnBuilder, d


def test_resolve_hit():
    test_attack = AttackBuilder(d(10))\
        .prof(3)\
        .amod(3)\
        .adv()\
        .gwm()\
        .attbon(3)\
        .dmgbon(2)\
        .build()
    assert 0.6525 == test_attack.resolve_hit(15)


def test_lucky_hit():
    test_attack = AttackBuilder(d(1)).adv().lucky().build()
    hit_outcome = test_attack.resolve_hit(15)
    assert 0.10212500000000002 == hit_outcome.p(HitOutcome.CRITICAL_HIT)


def test_resolve_attack():
    assert 0.04861111111111111 == AttackBuilder(d(6)).resolve(15).p(6)


def test_resolve_crit_attack():
    assert 1.0000000000000002 == AttackBuilder(d(1))\
        .attbon(-20)\
        .crit(0)\
        .resolve(0)\
        .p(2)


def test_resolve_turn_attacks():
    damage = TurnBuilder()\
        .attack(AttackBuilder(d(10))
                .prof(3)
                .amod(3)
                .adv()
                .gwm()
                .attbon(3)
                .dmgbon(2), times=2)\
        .attack(AttackBuilder(d(4))
                .prof(3)
                .amod(3)
                .adv()
                .gwm()
                .attbon(3)
                .dmgbon(2))\
        .dmgroll(d(2, 8))\
        .resolve(15)
    damage_stats = damage.stats()
    assert 52.81875000000001 == damage_stats.mean
