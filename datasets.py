import torch 
import torch.cuda
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

import functools
import csv
import copy
import glob
import os
import random

from collections import namedtuple

from utils.utils import xyz2irc, XyzTuple, IrcTuple
from utils.disk import getCache
from utils.logconfig import logging


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Disk cache path
raw_cache = getCache('diskCache')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    ['isNodule', 'diameter_mm', 'series_uid', 'center_xyz']
)


@functools.lru_cache(1)
def getCandidateInfoSet(requireOnDisk: bool=True) -> list[CandidateInfoTuple]: 
    # Merge candidates and annotations together in a readable format
    mhd_list = glob.glob('') #TODO
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # Extract data from annotations.csv
    diameter_dict = {}
    with open('') as f: #TODO
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple(float(x) for x in row[1:4])
            annotationDiameter_mm = float(row[4])
            isMalignant = bool(int(row[5]))

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    # Merge annotations and candidates data together
    candidates_list = []
    with open('') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # Check that series ID is already loaded on disk
            if series_uid not in presentOnDisk_set and requireOnDisk:
                continue

            candidateCenter_xyz = tuple(float(x) for x in row[1:4])
            isNodule = bool(int(row[4]))

            # Add diameter from annotations if center difference is small
            candidateDiameter_mm = 0.0
            for annotation_tuple in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tuple
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz - annotationCenter_xyz)
                    if delta_mm <= annotationCenter_xyz / 4:
                        candidateDiameter_mm = annotationDiameter_mm
                    else:
                        break
            
            candidates_list.append(CandidateInfoTuple(
                isNodule,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz
            ))

    candidates_list.sort(reverse=True)
    return candidates_list


# Function wrappers using cached memory
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid: 'seriesID', center_xyz: XyzTuple, width_irc: IrcTuple) -> (np.array, IrcTuple):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class Ct:
    def __init__(self, series_uid: 'seriesID') -> None:
        mhd_path = glob.glob(
            'data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid)     #TODO
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # Clip values in array to ignore random things like bones 
        ct_a.clip(ct_a, -1000, 1000)

        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz: XyzTuple, width_irc: IrcTuple) -> (np.array, IrcTuple):
        # Slice a chunk that contains the candidate nodule
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], \
            repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self, 
                val_stride: int = 0, 
                isValSet_bool: bool = False, 
                series_uid: 'SeriesID' = None,
                sortby_str: str =' None') -> None:
        # Splits data into validation and training selecting validation points based on stride
        # Taking copy of return value so that list in memory cache is not impacted by changes in this function
        self.candidateInfo_list = copy.copy(getCandidateInfoSet())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        
        # Creating validation set
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        # Creating training set
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        
        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self) -> int:
        return len(self.candidateInfo_list)

    def __getitem__(self, index: int) -> tuple:
        candidateInfo_tup = self.candidateInfo_list[index]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc
        )
        
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        # Two elements corresponding to nodule or non-nodule
        # nn.CrossEntropyLoss expects one output value per class
        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool,    # Value for not nodule
                candidateInfo_tup.isNodule_bool         # Value for nodule
            ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc)
        )









    

    












