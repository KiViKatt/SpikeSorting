from __future__ import annotations

# most important extractor are in spikeinterface.core
from spikeinterface.core import (
    BinaryFolderRecording,
    BinaryRecordingExtractor,
    NumpyRecording,
    NpzSortingExtractor,
    NumpySorting,
    NpySnippetsExtractor,
    ZarrRecordingExtractor,
    ZarrSortingExtractor,
    )

# sorting/recording/event from neo
from .neoextractors import *

# non-NEO objects implemented in neo folder
from .neoextractors import NeuroScopeSortingExtractor, MaxwellEventExtractor

# NWB sorting/recording/event
from .nwbextractors import NwbRecordingExtractor, NwbSortingExtractor

from .cbin_ibl import CompressedBinaryIblExtractor
from .iblextractors import IblRecordingExtractor, IblSortingExtractor
from .mcsh5extractors import MCSH5RecordingExtractor

# sorting extractors in relation with a sorter
from .klustaextractors import KlustaSortingExtractor
from .hdsortextractors import HDSortSortingExtractor
from .mclustextractors import MClustSortingExtractor
from .waveclustextractors import WaveClusSortingExtractor
from .yassextractors import YassSortingExtractor
from .combinatoextractors import CombinatoSortingExtractor
from .tridesclousextractors import TridesclousSortingExtractor
from .spykingcircusextractors import SpykingCircusSortingExtractor
from .herdingspikesextractors import HerdingspikesSortingExtractor
from .mdaextractors import MdaRecordingExtractor, MdaSortingExtractor
from .phykilosortextractors import PhySortingExtractor, KiloSortSortingExtractor
from .sinapsrecordingextractors import (
    SinapsResearchPlatformRecordingExtractor,
    )

# sorting in relation with simulator
from .shybridextractors import (
    SHYBRIDRecordingExtractor,
    SHYBRIDSortingExtractor,
    )

# snippers
from .waveclussnippetstextractors import WaveClusSnippetsExtractor


# misc
from spikeinterface.extractors.alfsortingextractor import ALFSortingExtractor


########################################

recording_extractor_full_list = [
    BinaryFolderRecording,
    BinaryRecordingExtractor,
    ZarrRecordingExtractor,
    # natively implemented in spikeinterface.extractors
    NumpyRecording,
    SHYBRIDRecordingExtractor,
    MdaRecordingExtractor,
    NwbRecordingExtractor,
    # others
    CompressedBinaryIblExtractor,
    IblRecordingExtractor,
    MCSH5RecordingExtractor,
    SinapsResearchPlatformRecordingExtractor,
]
recording_extractor_full_list += neo_recording_extractors_list

sorting_extractor_full_list = [
    NpzSortingExtractor,
    ZarrSortingExtractor,
    NumpySorting,
    # natively implemented in spikeinterface.extractors
    MdaSortingExtractor,
    SHYBRIDSortingExtractor,
    ALFSortingExtractor,
    KlustaSortingExtractor,
    HDSortSortingExtractor,
    MClustSortingExtractor,
    WaveClusSortingExtractor,
    YassSortingExtractor,
    CombinatoSortingExtractor,
    TridesclousSortingExtractor,
    SpykingCircusSortingExtractor,
    HerdingspikesSortingExtractor,
    KiloSortSortingExtractor,
    PhySortingExtractor,
    NwbSortingExtractor,
    NeuroScopeSortingExtractor,
    IblSortingExtractor,
]
sorting_extractor_full_list += neo_sorting_extractors_list

event_extractor_full_list = [MaxwellEventExtractor]
event_extractor_full_list += neo_event_extractors_list

snippets_extractor_full_list = [NpySnippetsExtractor, WaveClusSnippetsExtractor]

recording_extractor_full_dict = {}
for rec_class in recording_extractor_full_list:
    # here we get the class name, remove "Recording" and "Extractor" and make it lower case
    rec_class_name = rec_class.__name__.replace("Recording", "").replace("Extractor", "").lower()
    recording_extractor_full_dict[rec_class_name] = rec_class

sorting_extractor_full_dict = {}
for sort_class in sorting_extractor_full_list:
    # here we get the class name, remove "Extractor" and make it lower case
    sort_class_name = sort_class.__name__.replace("Sorting", "").replace("Extractor", "").lower()
    sorting_extractor_full_dict[sort_class_name] = sort_class

event_extractor_full_dict = {}
for event_class in event_extractor_full_list:
    # here we get the class name, remove "Extractor" and make it lower case
    event_class_name = event_class.__name__.replace("Event", "").replace("Extractor", "").lower()
    event_extractor_full_dict[event_class_name] = event_class
