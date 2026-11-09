"""serve.py request validation (no model, no network)."""
import serve


def setup_module():
    serve.presets.clear()
    serve.presets.update({"evil-recipe": {"spec": "14:8949:1"}, "golden-gate": {"spec": "6:7448:-0.8"}})


def body(**kw):
    return {"messages": [{"role": "user", "content": "hi"}], **kw}


def test_known_preset_passes_through():
    assert serve.parse_request(body(preset="golden-gate"))["preset"] == "golden-gate"


def test_unknown_or_missing_preset_falls_back_to_the_default():
    assert serve.parse_request(body(preset="rm -rf /"))["preset"] == serve.DEFAULT_PRESET
    assert serve.parse_request(body())["preset"] == serve.DEFAULT_PRESET


def test_numbers_are_clamped():
    r = serve.parse_request(body(dose=9, temperature=-1, max_new=99999))
    assert (r["dose"], r["temperature"], r["max_new"]) == (3.0, 0.0, 1024)


def test_history_is_capped_and_coerced_to_text():
    msgs = [{"role": "user", "content": 7}] * 50
    r = serve.parse_request({"messages": msgs})
    assert len(r["messages"]) == 40 and r["messages"][0]["content"] == "7"


def test_custom_spec_from_the_atlas():
    assert serve.parse_request(body(spec="14:8949:1, 5:123:-0.5"))["preset"] == "custom:14:8949:1,5:123:-0.5"
    assert serve.parse_request(body(spec="14:8949:1", preset="golden-gate"))["preset"].startswith("custom:")


def test_custom_spec_rejects_what_the_server_cannot_or_should_not_run():
    for bad in ("f9:7620:0.5", "32:1:1", "1:9216:1", "1:1:5", "1:1:0", "1:1", "a:b:c",
                ",".join(["1:1:0.1"] * 7), "", "1:1:nan"):
        assert serve.custom_spec(bad) is None, bad
    assert serve.parse_request(body(spec="f9:7620:0.5", preset="golden-gate"))["preset"] == "golden-gate"
