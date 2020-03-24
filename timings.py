# Used for profiling the polarization module

import polarization

if __name__ == "__main__":
    num_agents = polarization.NUM_AGENTS
    num_bins = polarization.NUM_BINS

    polarization.run_simulation( \
        polarization.build_mild_beliefs(num_agents, polarization.LOW_POLE, polarization.HIGH_POLE, polarization.BELIEF_STEP), \
        polarization.build_inf_graph_2_groups_disconnected(num_agents, 0.5), \
        400, \
        num_bins, \
        polarization.Update.BACKFIRE, \
        polarization.CONFBIAS_DISCOUNT, \
        polarization.BACKFIRE_BELIEF_THRESHOLD, \
        polarization.BACKFIRE_INFLUENCE_THRESHOLD \
    )
