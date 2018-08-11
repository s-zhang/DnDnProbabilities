from .HitOutcome import HitOutcome


class AttackOutcome:
    def __init__(self, hit_outcome: HitOutcome, damage: int):
        self.hit_outcome = hit_outcome
        self.damage = damage

    def __eq__(self, other):
        return self.damage == other.damage and \
            self.hit_outcome == other.hit_outcome

    def __hash__(self):
        return hash((self.damage, self.hit_outcome))

    def __str__(self):
        return "[" + str(self.damage) + " from " + str(self.hit_outcome) + "]"
