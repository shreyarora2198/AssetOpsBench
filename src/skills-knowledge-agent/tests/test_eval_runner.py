"""eval_runner ablation entrypoints (no full evaluate_all)."""

def test_condition_b_forecast_includes_sensor_data_in_result():
    from eval_runner import run_condition_b

    out = run_condition_b("Forecast next week's condenser water flow for Chiller 9.")
    assert "sensor_data" in out["result"]
    assert "forecast" in out["result"]
    assert out["result"]["forecast"].get("source") == "mock"


def test_condition_c_disables_knowledge(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from eval_runner import run_condition_c

    out = run_condition_c("Why is Chiller 6 behaving abnormally?")
    assert "metrics" in out
    assert out["metrics"].get("deep_tsfm_invoked") is False


def test_condition_d_runs_without_error(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from eval_runner import run_condition_d

    out = run_condition_d("What sensors are available for Chiller 6?")
    assert "result" in out
    assert out["metrics"].get("deep_tsfm_invoked") is False
