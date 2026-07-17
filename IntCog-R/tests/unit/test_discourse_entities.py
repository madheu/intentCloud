from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.discourse_entities import DiscourseEntities, EntityType, Gender, Role

def test_pronoun_resolution_haiou():
    de = DiscourseEntities(); de.add_from_text('大海、天空、海鸥')
    r = de.resolve('它')
    assert r == '海鸥', f'expected 海鸥, got {r}'

def test_pronoun_resolution_xiaohong():
    de = DiscourseEntities(); de.add_from_text('小红和小明在聊天')
    r = de.resolve('她')
    assert r == '小红', f'expected 小红, got {r}'

def test_no_female_entity():
    de = DiscourseEntities(); de.add('小明', EntityType.PERSON, Gender.MALE)
    r = de.resolve('她')
    assert r is None, f'expected None, got {r}'

def test_agent_resolution():
    de = DiscourseEntities()
    de.add('小王', EntityType.PERSON, Gender.MALE, Role.AGENT)
    de.add('小李', EntityType.PERSON, Gender.MALE, Role.PATIENT)
    r = de.resolve('他')
    candidates = [e.name for e in de.entities if e.role == Role.AGENT]
    assert '小王' in candidates
    assert r in ('小王', '小李')

def test_recent_entity_preferred():
    de = DiscourseEntities(); de.add_from_text('小红和小明在聊天')
    de.add('小张', EntityType.PERSON, Gender.FEMALE)
    r = de.resolve('她')
    assert r == '小张', f'expected 小张, got {r}'

def test_clear():
    de = DiscourseEntities(); de.add('test'); de.clear()
    assert len(de.entities) == 0
