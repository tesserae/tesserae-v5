import pytest

import copy


def entity_init(entity, default_attrs, valid_attrs):
    e = entity()
    assert hasattr(e, '_id')
    assert e.__dict__ == default_attrs

    for v in valid_attrs:
        print(v)
        fix_id = False
        if '_id' in v:
            v['id'] = v.pop('_id')
            fix_id = True

        e = entity(**v)

        if 'id' in v or fix_id:
            v['_id'] = v.pop('id')

        for k in e.__dict__:
            assert k in v
            assert e.__dict__[k] == v[k]
            assert getattr(e, k) == v[k]


def entity_eq(entity, default_attrs, valid_attrs):
    for i in range(len(valid_attrs)):
        for j in range(len(valid_attrs)):
            e1 = entity(id=valid_attrs[i])
            e2 = entity(id=valid_attrs[j])

            if i != j:
                assert e1 != e2
            else:
                assert e1 == e2


def entity_copy(entity, default_attrs, valid_attrs):
    e1 = entity()
    e2 = e1.copy()
    assert e2 is not e1
    assert e2 == e1

    for v in valid_attrs:
        e1 = entity(id=v)
        e2 = e1.copy()
        assert e2 is not e1
        assert e2 == e1


def entity_id(entity, default_attrs, valid_attrs):
    e = entity()
    assert e.id == e._id

    for v in valid_attrs:
        e = entity(id=v['_id'])
        assert e.id == v['_id']
        assert e.id == e._id


def entity_json_encode(entity, default_attrs, valid_attrs):
    e = entity()
    assert e.json_encode() == default_attrs

    e = entity()
    assert e.json_encode(exclude=['foo']) == default_attrs

    attrs = list(default_attrs.keys())
    for i, k in enumerate(attrs):
        def_copy = copy.deepcopy(default_attrs)
        def_copy.pop(k)

        e = entity()
        assert e.json_encode(exclude=[k]) == def_copy
        assert e.json_encode(exclude=[k, 'foo']) == def_copy
        assert e.json_encode(exclude=['foo', k]) == def_copy

        def_copy = copy.deepcopy(default_attrs)
        excluded = ['foo']
        for k in attrs[:i]:
            excluded.append(k)
            def_copy.pop(k)

        assert e.json_encode(exclude=excluded) == def_copy
        assert e.json_encode(exclude=excluded + ['foo']) == def_copy
        assert e.json_encode(exclude=['foo'] + excluded) == def_copy

        def_copy = copy.deepcopy(default_attrs)
        excluded = ['foo']
        for k in attrs[i:]:
            excluded.append(k)
            def_copy.pop(k)

        assert e.json_encode(exclude=excluded) == def_copy
        assert e.json_encode(exclude=excluded + ['foo']) == def_copy
        assert e.json_encode(exclude=['foo'] + excluded) == def_copy

    for v in valid_attrs:
        fix_id = False
        if '_id' in v:
            v['id'] = v.pop('_id')
            fix_id = True

        e = entity(**v)

        if fix_id:
            v['_id'] = v.pop('id')

        for i, k in enumerate(attrs):
            def_copy = copy.deepcopy(v)
            def_copy.pop(k)

            assert e.json_encode(exclude=[k]) == def_copy
            assert e.json_encode(exclude=[k, 'foo']) == def_copy
            assert e.json_encode(exclude=['foo', k]) == def_copy

            def_copy = copy.deepcopy(v)
            excluded = ['foo']
            for k in attrs[:i]:
                excluded.append(k)
                def_copy.pop(k)

            assert e.json_encode(exclude=excluded) == def_copy
            assert e.json_encode(exclude=excluded + ['foo']) == def_copy
            assert e.json_encode(exclude=['foo'] + excluded) == def_copy

            def_copy = copy.deepcopy(v)
            excluded = ['foo']
            for k in attrs[i:]:
                excluded.append(k)
                def_copy.pop(k)

            assert e.json_encode(exclude=excluded) == def_copy
            assert e.json_encode(exclude=excluded + ['foo']) == def_copy
            assert e.json_encode(exclude=['foo'] + excluded) == def_copy


def entity_json_decode(entity, default_attrs, valid_attrs):
    e = entity()
    decoded = entity.json_decode({})
    assert decoded == e

    for v in valid_attrs:
        fix_id = False
        if '_id' in v:
            v['id'] = v.pop('_id')
            fix_id = True

        e = entity(**v)

        if fix_id:
            v['_id'] = v.pop('id')

        decoded = entity.json_decode(v)
        assert decoded == e
