import logging
from typing import Sequence, Union

from .entity import Entity
from .merge import Merge
from .transform import Transform

import numpy as np

logger = logging.getLogger(__name__)

"""
Contains classes which define configuration used for transforming and modifying objects, as well as the associated
validation routines.  Ideally, a configuration class should be defined for every pipeline that is defined.
"""


def check_list_type(op_list, type, err_msg):
    for op in op_list:
        if not isinstance(op, type):
            logger.error(err_msg)
            raise ValueError(err_msg)


class XFormMergePipelineConfig:
    """
    Defines all configuration items necessary to run the XFormMerge Pipeline, and associated configuration validation.

    NOTE: the argument list can be condensed into lists of lists, but that becomes a bit less intuitive to use.  We
    need to think about how best we want to specify these argument lists.
    """
    def __init__(self, trigger_list: Sequence[Entity] = None, trigger_sampling_prob: Sequence[float] = None,
                 trigger_xforms: Sequence[Transform] = None, trigger_bg_xforms: Sequence[Transform] = None,
                 trigger_bg_merge: Merge = None, trigger_bg_merge_xforms: Sequence[Transform] = None,
                 merge_type: str = 'insert',
                 per_class_trigger_frac: float = None):
        """
        Initializes the configuration used by XFormMergePipeline
        :param trigger_list: a list of Triggers to insert into the background Entity
        :param trigger_sampling_prob: probability with how the trigger should be sampled, if none, uniform sampling
                                      happens
        :param trigger_xforms: a list of transforms to apply to the trigger
        :param trigger_bg_xforms: a list of transforms to apply to the trigger background (what the trigger will be
                                  inserted into)
        :param trigger_bg_merge: merge operator to combine the trigger and the trigger background
        :param trigger_bg_merge_xforms: a list transforms to apply after combining the trigger and the trigger
                                        background
        :param merge_type: How data will be merged.  Valid merge_types are determined by the method argument of the
                           Pipeline's modify_clean_dataset() function
        :param per_class_trigger_frac: The percentage of the total clean data to modify.  If None, all the data will
                                       be modified
        """
        self.trigger_list = trigger_list
        self.trigger_xforms = trigger_xforms
        self.trigger_sampling_prob = trigger_sampling_prob

        self.trigger_bg_xforms = trigger_bg_xforms
        self.trigger_bg_merge = trigger_bg_merge
        self.trigger_bg_merge_xforms = trigger_bg_merge_xforms

        # validate configuration based on the merge type
        self.merge_type = merge_type.lower()
        self.per_class_trigger_frac = per_class_trigger_frac

        self.validate()

    def validate(self):
        """
        Validates whether the configuration was setup properly, based on the merge_type.
        :return: None
        """

        if self.per_class_trigger_frac is not None and (self.per_class_trigger_frac <= 0. or
                                                        self.per_class_trigger_frac >= 1.):
            msg = "per_class_trigger_frac must be between 0 and 1, noninclusive"
            logger.error(msg)
            raise ValueError(msg)

        if self.merge_type == 'insert' or self.merge_type == 'regenerate':
            pass
        else:
            msg = "Unknown merge_type! See pipeline's modify_clean_dataset() for valid merge types!"
            logger.error(msg)
            raise ValueError(msg)

        # the following set of variables are
        if self.trigger_list is None:
            msg = "No triggers specified to be inserted!"
            logger.error(msg)
            raise ValueError(msg)
        check_list_type(self.trigger_list, Entity, "trigger_list must be a sequence of Entity objects!")
        if self.trigger_sampling_prob is not None:
            check_list_type(self.trigger_sampling_prob, float, "trigger_sampling_prob must be a sequence of floats!")

        if self.trigger_xforms is None:
            # silently convert None to no xforms applied in the format needed by the Pipeline
            self.trigger_xforms = []
        check_list_type(self.trigger_xforms, Transform, "trigger_xforms must be a list of Transform objects!")

        if self.trigger_bg_merge is None or not isinstance(self.trigger_bg_merge, Merge):
            msg = "trigger_bg_merge must be specified as a trojai.datagen.Merge.Merge object"
            logger.error(msg)
            raise ValueError(msg)

        if self.trigger_bg_merge_xforms is None:
            # silently convert None to no xforms applied in the format needed by the Pipeline
            self.trigger_bg_merge_xforms = []
        check_list_type(self.trigger_bg_merge_xforms, Transform,
                        "trigger_bg_merge_xforms must be a list of Transform objects")


class InsertAtRandomLocationConfig:
    """
    Specifies which algorithm to use for determining the valid spots for trigger insertion on an image and all
    relevant parameters
    """

    def __init__(self, algorithm: str = 'brute_force', min_val: Union[int, Sequence[int]] = 0,
                 threshold_val: Union[float, Sequence[float]] = 5.0, num_boxes: int = 5):
        """
        Initialize and validate all relevant parameters for InsertAtRandomLocation
        :param algorithm: algorithm to use for determining valid placement, options include
        brute_force -> for every edge pixel of the image, invalidates all intersecting pattern insert locations
        threshold -> a trigger position on the image is invalid if the mean pixel value over the area is greater than a
                     specified amount (threshold_val),
                     WARNING: slowest of all options by substantial amount
        edge_tracing -> follows perimeter of non-zero image values invalidating locations where there is any overlap
                        between trigger and image, works well for convex images with long, flat edges
        bounding_boxes -> splits the image into a grid of size num_boxes x num_boxes and generates a bounding box for
                          the image in each grid location, and invalidates all intersecting trigger insert locations,
                          provides substantial speedup for large images with fine details but will not find all valid
                          insert locations,
                          WARNING: may not find valid any valid insert locations if image or num_boxes are too small
        :param min_val: any pixels above this value will be considered for determining overlap, any below this value
                        will be treated as if there is no image present for the given pixel
        :param threshold_val: value to compare mean pixel value over possible insert area to,
                              only needed for threshold
        :param num_boxes: size of grid for bounding boxes algorithm, larger value implies closer approximation,
                          only needed for bounding_boxes
        """
        self.algorithm = algorithm.lower()
        self.min_val = min_val
        self.threshold_val = threshold_val
        self.num_boxes = num_boxes

        self.validate()

    def validate(self):
        """
        Assess validity of provided values
        :return: None
        """

        if self.algorithm not in {'brute_force', 'threshold', 'edge_tracing', 'bounding_boxes'}:
            msg = "Algorithm specified is not implemented!"
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(self.min_val, (int, float)):
            if self.min_val < 0.0:
                msg = "Illegal value specified for min_val.  Must be non-negative!"
                logger.error(msg)
                raise ValueError(msg)
        elif isinstance(self.min_val, (tuple, list, np.ndarray)):
            for val in self.min_val:
                if val < 0.0:
                    msg = "Illegal value specified for min_val.  All values must be non-negative!"
                    logger.error(msg)
                    raise ValueError(msg)
        else:
            msg = "Illegal type for min_val.  Must either be a scalar or a tuple of min values per channel"
            logger.error(msg)
            raise TypeError(msg)

        if self.algorithm == 'brute_force':
            pass

        elif self.algorithm == 'threshold':
            if isinstance(self.threshold_val, (int, float)):
                if self.threshold_val < 0.0:
                    msg = "Illegal value specified for threshold_val.  Must be non-negative!"
                    logger.error(msg)
                    raise ValueError(msg)
            elif isinstance(self.threshold_val, (tuple, list, np.ndarray)):
                for val in self.threshold_val:
                    if val < 0.0:
                        msg = "Illegal value specified for threshold_val.  All values must be non-negative!"
                        logger.error(msg)
                        raise ValueError(msg)
            else:
                msg = "Illegal type for threshold_val.  " \
                      "Must either be a scalar or a tuple of threshold values per channel!"
                logger.error(msg)
                raise TypeError(msg)

        elif self.algorithm == 'edge_tracing':
            pass

        elif self.algorithm == 'bounding_boxes':
            if self.num_boxes < 1 or self.num_boxes > 25:
                msg = "Must specify a value between 1 and 25 for num_boxes!"
                logger.error(msg)
                raise ValueError(msg)
