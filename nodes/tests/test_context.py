def test_context_get_parameter_global(context, parameter):
    context.add_global_parameter(parameter)
    assert parameter.global_id == parameter.local_id
    assert context.get_parameter(parameter.global_id) == parameter


def test_context_get_parameter_local(context, node, parameter):
    node.add_parameter(parameter)
    assert parameter.global_id != parameter.local_id
    assert context.get_parameter(parameter.global_id) == parameter


def test_context_activate_scenario_sets_active_scenario(context, scenario):
    assert context.active_scenario != scenario
    context.activate_scenario(scenario)
    assert context.active_scenario == scenario
