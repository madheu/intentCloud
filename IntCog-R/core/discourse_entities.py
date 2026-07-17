# H17 R1: 指代消解
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class EntityType(Enum): PERSON = 'person'; OBJECT = 'object'; EVENT = 'event'
class Gender(Enum): MALE = 'male'; FEMALE = 'female'; NEUTER = 'neuter'; UNKNOWN = 'unknown'
class Role(Enum): AGENT = 'agent'; PATIENT = 'patient'; BENEFICIARY = 'beneficiary'; BYSTANDER = 'bystander'; UNKNOWN = 'unknown'

PRONOUN_MAP = {
    '他': (Gender.MALE, EntityType.PERSON), '她': (Gender.FEMALE, EntityType.PERSON),
    '它': (Gender.NEUTER, EntityType.OBJECT), '这': None, '那': None,
    '谁': (Gender.UNKNOWN, EntityType.PERSON), '怎么': None, '什么': None,
    '他们': (Gender.MALE, EntityType.PERSON), '她们': (Gender.FEMALE, EntityType.PERSON), '它们': (Gender.NEUTER, EntityType.OBJECT),
}

@dataclass
class DiscourseEntity:
    name: str; etype: EntityType = EntityType.OBJECT
    gender: Gender = Gender.NEUTER; role: Role = Role.UNKNOWN
    mention_count: int = 1; last_mention_turn: int = 0

class DiscourseEntities:
    def __init__(self): self.entities: list[DiscourseEntity] = []; self._turn = 0

    def add(self, name: str, etype=EntityType.OBJECT, gender=Gender.NEUTER, role=Role.UNKNOWN):
        self._turn += 1
        for e in self.entities:
            if e.name == name: e.mention_count += 1; e.last_mention_turn = self._turn; return e
        e = DiscourseEntity(name=name, etype=etype, gender=gender, role=role, last_mention_turn=self._turn)
        self.entities.append(e); return e

    def add_from_text(self, text: str):
        known = {'小红':('person','female'),'小明':('person','male'),'小王':('person','male'),'小李':('person','male'),
                 '大海':('object','neuter'),'天空':('object','neuter'),'海鸥':('object','neuter')}
        for name, (t,g) in known.items():
            if name in text:
                et = EntityType.PERSON if t=='person' else EntityType.OBJECT
                gn = Gender.FEMALE if g=='female' else Gender.MALE if g=='male' else Gender.NEUTER
                self.add(name, et, gn)

    def resolve(self, pronoun: str) -> Optional[str]:
        info = PRONOUN_MAP.get(pronoun)
        candidates = self.entities[:]
        if info:
            target_gender, target_type = info
            candidates = [e for e in candidates if target_type is None or e.etype == target_type]
            candidates = [e for e in candidates if target_gender is None or e.gender == target_gender or e.gender == Gender.UNKNOWN]
            if pronoun in ('这',): candidates = sorted(candidates, key=lambda e: e.last_mention_turn, reverse=True)
            elif pronoun in ('那',): candidates = [e for e in candidates if e.last_mention_turn < self._turn - 2]
        candidates.sort(key=lambda e: (e.mention_count, e.last_mention_turn), reverse=True)
        return candidates[0].name if candidates else None

    def get_role(self, name: str) -> Role:
        for e in self.entities:
            if e.name == name: return e.role
        return Role.UNKNOWN

    def clear(self): self.entities.clear(); self._turn = 0
    def next_turn(self): self._turn += 1
