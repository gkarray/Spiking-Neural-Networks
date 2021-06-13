import nengo
import numpy as np

LEARNING_RATE = 1e-4
TIMESTEP = 0.001

def NengoModelCartpole(environment):
    model = nengo.Network()

    with model:
        state_node = nengo.Node(environment.get_state, size_out=4, label='N-State')
        reward_node = nengo.Node(environment.get_reward, size_out=1, label='N-Reward')
        action_node = nengo.Node(environment.set_action, size_in=2, label='N-Action')

        state = nengo.Ensemble(n_neurons=3000, dimensions=4, radius=10, label='State') 
        nengo.Connection(state_node, state, synapse=0)

        reward = nengo.Ensemble(n_neurons=1000, dimensions=1, radius=1, label='Reward')
        nengo.Connection(reward_node, reward, synapse=0)

        q_value = nengo.Ensemble(n_neurons=4000, dimensions=2, radius=25, label='Q-Value')

        learning_connection = nengo.Connection(state, q_value, 
                                        function = lambda x:[0, 0],
                                        learning_rule_type = nengo.PES(LEARNING_RATE),
                                        synapse = 0)

        BG = nengo.networks.BasalGanglia(dimensions=2,n_neurons_per_ensemble=1000)
        nengo.Connection(q_value, BG.input)

        thalamus = nengo.networks.Thalamus(dimensions=2, n_neurons_per_ensemble=1000)
        nengo.Connection(BG.input, thalamus.input)

        old_q_stage_1 = nengo.Ensemble(n_neurons = 4000, dimensions=4, radius=25, label='old Q stage 1')
        nengo.Connection(thalamus.output, old_q_stage_1[:2], synapse=0)
        nengo.Connection(q_value, old_q_stage_1[2:], synapse=0)

        def old_q_func(x):
            return x[:2] * x[2:]

        old_q = nengo.Ensemble(n_neurons =4000, dimensions=2, radius=25, label='old Q')
        nengo.Connection(old_q_stage_1, old_q, function=old_q_func)

        new_q_stage_1 = nengo.Ensemble(n_neurons = 4000, dimensions=4, radius=25, label='new Q stage 1')
        nengo.Connection(thalamus.output, new_q_stage_1[:2], synapse=0)
        nengo.Connection(q_value, new_q_stage_1[2:], synapse=0)

        def new_q_func(x):
            return np.max(x[:2] * x[2:])

        new_q = nengo.Ensemble(n_neurons = 4000, dimensions=1, radius=25, label='new Q')
        nengo.Connection(new_q_stage_1, new_q, function=new_q_func)

        td_error_stage_1 = nengo.Ensemble(n_neurons = 4000, dimensions=4, radius = 25, label='TD-error stage 1')
        nengo.Connection(new_q, td_error_stage_1[0], synapse = TIMESTEP)
        nengo.Connection(reward, td_error_stage_1[1], synapse=TIMESTEP)
        nengo.Connection(old_q, td_error_stage_1[2:], synapse=TIMESTEP*(environment.update_each))

        def td_error_func(x):
            ones_encoding = np.array([0, 0])
            ones_encoding[np.argmax(x[2:])] = 1
            td = -x[1] * ones_encoding - 0.9 * ones_encoding * x[0] + ones_encoding * x[2:]
            print(td)
            return td

        td_error = nengo.Ensemble(n_neurons=4000, dimensions = 2, radius= 3, label='TD-error')
        nengo.Connection(td_error_stage_1, td_error, function=td_error_func, synapse=0)

        nengo.Connection(td_error, learning_connection.learning_rule, synapse=0)


        nengo.Connection(thalamus.output, action_node, synapse=0)

    return model
