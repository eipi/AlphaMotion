from collections import namedtuple


SampleBase = namedtuple("SampleBase", ["start", "end"])
class Sample(SampleBase):
    def __new__(cls, start, end):
        obj = SampleBase.__new__(cls, start, end)
        return obj


def filter_dominant_classes_only(classification_weighting_map):
    filtered_classifications = {}
    for sample in classification_weighting_map.keys():
        max_weighting = 0
        weightings = classification_weighting_map.get(sample)
        for classification in weightings.keys():
            if weightings.get(classification) >= max_weighting:
                max_weighting = weightings.get(classification)
                filtered_classifications[sample] = classification
    return filtered_classifications


def find_dominant_class_for_samples(labels_index, full_sample_sequence):
    class_map = {}
    running_sample_index = 0
    for classification_sequence in labels_index:
        while full_sample_sequence[running_sample_index][1] < classification_sequence[1]:
            current_sample_raw = full_sample_sequence[running_sample_index]
            current_sample = Sample(current_sample_raw[0], current_sample_raw[1])
            current_sample_class_weighting = current_sample.end - max(classification_sequence[0], current_sample.start) + 1
            if current_sample in class_map.keys():
                existing_classifications = class_map.get(current_sample)
                existing_classifications[classification_sequence[2]] = current_sample_class_weighting
            else:
                class_map[current_sample] = {classification_sequence[2]: current_sample_class_weighting}
            running_sample_index = running_sample_index + 1
        if current_sample.end >= classification_sequence[1]:
            class_map[current_sample] = {classification_sequence[2]: classification_sequence[1] - current_sample.start + 1}
        if len(full_sample_sequence) > running_sample_index:
            next_sample = Sample(full_sample_sequence[running_sample_index][0], full_sample_sequence[running_sample_index][1])
            if next_sample.start < classification_sequence[1]:
                class_map[next_sample] = {classification_sequence[2]: classification_sequence[1] - next_sample.start + 1}
    #print(class_map)
    filtered_classmap = filter_dominant_classes_only(class_map)
    #print(filtered_classmap)
    return filtered_classmap

