# Copyright (C) 2017 Chenxi Shan and Zhixian Ma

"""
A class to make statistics on the WCA and ACA competition results.

Keys
====
competitionId
eventId
roundTypeId
pos
best
average
personName
personId
personCountryId
formatId
value1, value2, value3, value4, value5

event_keys
==========
222,
333, 333bf, 333fm, 333ft, 333mbf, 333mbo, 333oh,
444, 444bf,
555, 555bf,
666,
777,
clock, magic, minx mmagic, pyram, skewb, sq1

methods
=======
load_table:
    load the TSV or CSV style result table
gen_pool:
    get record pool of a single event w.r.t. to the pool type
get_estimate:
    Estimate the normal-distributed parameters w.r.t. to the event from its pool
gen_rank:
    Calculate ranks w.r.t. the estimated paramters by get_estimate
get_evaluation:
    Evaluate rank of persons of a single event
get_evaluation_all:
    Evaluate rank of persons of all of the events
save_rank:
    Save the ranks
save_evaluation_result:
    Save results of persons
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import os


class ACA():
    """The ACA class"""
    def __init__(self, tablepath=None):
        self.tablepath = tablepath
        self.table = self.load_list()
        self.keys_result = ("CompetitionId", "eventId", "ropeundTypeId",
                            "pos","best","average","personName","personId",
                            "personCountryId","formatId","value1",
                            "value2","value3","value4","value5")
        self.samples = len(self.table)
        self.keys_event = ("222","333","333bf","333fm","333ft","333mbf","333oh",
                           "444","444bf","555","555bf","666","777","clock",
                           "magic","minx","mmagic","pyram","skewb","sq1")
        self.est_grp = {"single": ("value1","value2","value3","value4","value5"),
                        "average": ("average"),
                        "best": ("best")}

    def load_list(self):
        """load the competition result table."""
        try:
            table = pd.read_csv(self.tablepath, sep="\t")
        except:
            print("The result table cannot be loaded.")
            return None

        return table


    def gen_pool(self,event_key,est_grp):
        """Generate pool of a single event,

        inputs
        ======
        event_key: str
            Key of the event, should be in the self.keys_event
        est_grp: strp
            Key of the values to be pooled, should be in self.est_grp

        output
        ======
        pool: dict
            {"event_key": event_key, "est_grp": est_grp, "value": value}
        """
        # Judge keys
        if event_key not in self.keys_event:
            print("The event %s is a wrong event." % (event_key))
            return None

        if est_grp not in self.est_grp.keys():
            print("The group %s is a wrong group." % (est_grp))
            return None

        # generate the pool
        sample_idx = np.where(self.table["eventId"] == event_key)[0]
        if est_grp == "single":
            value_list = []
            for key in self.est_grp[est_grp]:
                value_list.append(self.table[key][sample_idx])
            # stack
            value = np.hstack(value_list)
        else:
            value = self.table[self.est_grp[est_grp]][sample_idx]

        # pooldict
        pool = {"event_key": event_key,
                "est_grp": est_grp,
                "value": np.array(value)}

        return pool


    def get_estimate(self, pool):
        """
        Estimate the normal-distributed parameters w.r.t.
        the event from its pool.

        Reference
        =========
        [1] histogram
            np.histogram
        [2] scipy.stats.norm
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

        input
        =====
        pool: dict
            The pooled sample results.

        output
        ======
        params: tuple
            The estimated normal distribution paramerts, (miu, sigma).
        """
        # estimate
        miu, sigma = norm.fit(pool["value"])
        # estimator
        rv = norm(loc=miu, scale=sigma)
        # update
        pool.update({"params":(miu,sigma),
                     "rv":rv})

        return pool


