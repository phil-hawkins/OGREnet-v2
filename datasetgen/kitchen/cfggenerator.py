import json
import copy

from scene import Scene, SceneObject

# Selection: holds the set of terminal symbols produced by production rules
# and a list of indexes of scene objects that matches the text
class Selection:
    def __init__(self, text='', scene_object_indexs=[], tags=[], canonical_text=None, scene_number=None):
        self.text = text
        self.scene_object_indexs = scene_object_indexs
        self.tags = tags
        self.canonical_text = canonical_text
        self.scene_number = scene_number
    
    def to_string(self):
        return '{}\nselection: {}\ntags: {}\n'.format(self.text, self.scene_object_indexs, self.tags)

# SequenceInstance: represents a production rule sequnce as a list of Selections
# each Selection object maps to a symbol in the sequence
class SequenceInstance:
    def __init__(self, selections=[]):
        self.selections = selections

    # join the symbols together into a single selection for the sequence
    def join(self, scene_object_indexs=[], tags=[]):
        result = Selection()
        all_tags = [tags]
        for selection in self.selections:
            if result.text != '':
                result.text += ' '
            result.text += selection.text
            all_tags.append(selection.tags)
        result.scene_object_indexs = scene_object_indexs
        result.tags = list(set().union(*all_tags))

        if len(self.selections) == 1:
            result.canonical_text = self.selections[0].canonical_text

        return result 

class ProductionRule:        
    def __init__(self, symbol, expansion, filtertype=None, min=None, max=None, passThroughSymbolSelection=None, tags=[], canonical_text=None):
        self.symbol = symbol
        self.expansion = expansion
        self.filtertype = filtertype
        self.min = min
        self.max = max
        self.passThroughSymbolSelection = passThroughSymbolSelection
        self.tags = tags
        self.canonical_text = canonical_text

    @classmethod
    def from_dict(cls, dict):
        if 'symbol' in dict:
            pr = ProductionRule(
                symbol = dict['symbol'],
                expansion = dict['expansion'],
                filtertype = dict['filtertype'] if 'filtertype' in dict.keys() else None,
                min = dict['min'] if 'min' in dict.keys() else None,
                max = dict['max'] if 'max' in dict.keys() else None,
                passThroughSymbolSelection = dict['passThroughSymbolSelection'] if 'passThroughSymbolSelection' in dict.keys() else None,
                tags = str.split(dict['tags']) if 'tags' in dict.keys() else [],
                canonical_text = dict['canonical_text'] if 'canonical_text' in dict.keys() else None
            )
            return pr
        else:
            return dict

    def to_dict(self):
        return {
            'symbol' : self.symbol,
            'expansion' : self.expansion,
            'filtertype' : self.filtertype,
            'min' : self.min,
            'max' : self.max,
            'passThroughSymbolSelection' : self.passThroughSymbolSelection,
            'tags' : ' '.join(self.tags),
            'canonical_text' : self.canonical_text
        }

    def select_scene_object(self, sequence_instances, scene):
        assert(len(sequence_instances) == 1)
        sequence_instance = sequence_instances[0]
        assert(len(sequence_instance.selections) == 1)
        selection = sequence_instance.selections[0]
        selection.tags = self.tags

        assert(selection.canonical_text is not None)
        filter_text = selection.canonical_text

        scene_object_indexes = range(len(scene.all_objects()))
        selection.scene_object_indexs = scene.get_selection(scene_object_indexes, SceneObject.is_scene_object_type, [filter_text])
        return [selection]

    def binary_compare(self, sequence_instances, scene):
        selections = []
        for sequence_instance in sequence_instances:
            assert(len(sequence_instance.selections) == 3)
            # first symbol represents the list of objects to select from
            object_list = sequence_instance.selections[0].scene_object_indexs

            # only proceed if the selection is non-trivial
            # e.g. "the fork left of the knife" when there is only one fork
            if (len(object_list) > 1):
                # second symbol is the relationship
                assert(sequence_instance.selections[1].canonical_text is not None)
                rel = sequence_instance.selections[1].canonical_text
                # third symbol represents the comparison object
                assert(len(sequence_instance.selections[2].scene_object_indexs) == 1)
                object_index = sequence_instance.selections[2].scene_object_indexs[0]
                comparison_object = scene.all_objects()[object_index]

                # filter the list of objects by relation to the comparison object
                args = (comparison_object, rel)
                selected_object_indexes = scene.get_selection(object_list, SceneObject.is_related, args)

                selections.append(sequence_instance.join(selected_object_indexes, self.tags))

        return selections

    def scene_compare(self, sequence_instances, scene):
        selections = []
        for sequence_instance in sequence_instances:
            assert(len(sequence_instance.selections) == 3)
            # first symbol represents the list of objects to select from
            object_list = sequence_instance.selections[0].scene_object_indexs
            objects = [scene.all_objects()[i] for i in object_list]

            # only proceed if the selection is non-trivial
            # e.g. "the fork left of the knife" when there is only one fork
            if (len(object_list) > 1):
                # second symbol is the relationship
                assert(sequence_instance.selections[1].canonical_text is not None)
                rel = sequence_instance.selections[1].canonical_text
                
                # third symbol represents the comparison object
                assert(len(sequence_instance.selections[2].scene_object_indexs) == 1)
                object_index = sequence_instance.selections[2].scene_object_indexs[0]
                comparison_object = scene.all_objects()[object_index]

                # filter the list of objects by relation to the comparison object with respect to the other scene objects of that type
                args = (comparison_object, rel, objects)
                selected_object_indexes = scene.get_selection(object_list, SceneObject.scene_compare_pred, args)

                selections.append(sequence_instance.join(selected_object_indexes, self.tags))

        return selections

    def between(self, sequence_instances, scene):
        selections = []
        for sequence_instance in sequence_instances:
            assert(len(sequence_instance.selections) == 5)
            # first symbol represents the list of objects to select from
            object_list = sequence_instance.selections[0].scene_object_indexs

            # only proceed if the selection is non-trivial
            if (len(object_list) > 1):
                # second symbol is 'between'
                # third symbol is the first bounding object
                assert(len(sequence_instance.selections[2].scene_object_indexs) == 1)
                object_index1 = sequence_instance.selections[2].scene_object_indexs[0]
                bounding_object1 = scene.all_objects()[object_index1]
                # fourth symbol is 'and'
                # fifth symbol is the second bounding object
                assert(len(sequence_instance.selections[4].scene_object_indexs) == 1)
                object_index2 = sequence_instance.selections[4].scene_object_indexs[0]
                bounding_object2 = scene.all_objects()[object_index2]

                if (object_index1 != object_index2):
                    args = (bounding_object1, bounding_object2)
                    selected_object_indexes = scene.get_selection(object_list, SceneObject.is_between_pred, args)
                    selected_object_indexes = [i for i in selected_object_indexes if i not in [object_index1, object_index2]]

                    selections.append(sequence_instance.join(selected_object_indexes, self.tags))

        return selections

    def prune_on_cardinality(self, selections):
        if self.min is not None:
            selections = [s for s in selections if len(s.scene_object_indexs) >= self.min]
        if self.max is not None:
            selections = [s for s in selections if len(s.scene_object_indexs) <= self.max]
        return selections

    def apply_filters(self, scene, sequence_instances):
        if self.filtertype == 'scene':
            expanded_sequence = self.select_scene_object(sequence_instances, scene)
        elif self.filtertype == 'binary_compare':
            expanded_sequence = self.binary_compare(sequence_instances, scene)
        elif self.filtertype == 'between':
            expanded_sequence = self.between(sequence_instances, scene)
        elif self.filtertype == 'scene_compare':
            expanded_sequence = self.scene_compare(sequence_instances, scene)
        elif self.filtertype == 'pass_through':
            assert(self.passThroughSymbolSelection is not None)
            expanded_sequence = []
            for si in sequence_instances: 
                s = si.join(si.selections[self.passThroughSymbolSelection].scene_object_indexs, self.tags)
                s.canonical_text = si.selections[self.passThroughSymbolSelection].canonical_text
                expanded_sequence.append(s)
        else:
            expanded_sequence = [si.join(tags=self.tags) for si in sequence_instances]

        return expanded_sequence


class CFGGenerator:
    @classmethod
    def isTerminal(cls, symbol):
        return symbol[0] != '_'

    def __init__(self, scene, production_rules=None):
        self.production_rules = production_rules
        self.scene = scene

    def add_rule(self, production_rule):
        self.production_rules[production_rule.symbol] = production_rule

    def vocab(self):
        vocab = set()
        for _, rule in self.production_rules.items():
            expansion = rule.expansion
            for sq in expansion:
                for s in sq:
                    if s[0] != '_':
                        for w in s.split():
                            vocab.add(w)
        return vocab


    # expand out a symbol into all the possible versions of the production rules for the symbol
    # the list is pruned for symbols the don't match the required cardinality 
    # of the production rule
    # e.g. if the symbol represts a single object, the production rule will
    # disallow results with empty sets or more than one match
    def expand_symbol(self, symbol):
        production_rule = self.production_rules[symbol]
        result = [] # list of Selection
        for sequence in production_rule.expansion:
            sequence_instances = self.expand_sequence(sequence, production_rule.canonical_text)
            selections = production_rule.apply_filters(self.scene, sequence_instances)
            selections = production_rule.prune_on_cardinality(selections)
            result += selections
        return result

    # returns a list of SequenceInstances one for each valid expansion of the sequence
    # each final sequence instance has a list of selections that maps to the symbols
    # of the sequence
    def expand_sequence(self, sequence, canonical_text):
        result = []

        if self.isTerminal(sequence[-1]):
            selection = Selection(text=sequence[-1], canonical_text=canonical_text)
            rhs = [selection]
        else:
            rhs = self.expand_symbol(sequence[-1])

        if len(sequence) == 1:
            result = [SequenceInstance(selections=[sel]) for sel in rhs]
        else:
            lhs = self.expand_sequence(sequence[:-1], canonical_text)
            for lhs_seq_instance in lhs:
                for rhs_selection in rhs:
                    seq_instance = copy.deepcopy(lhs_seq_instance)
                    seq_instance.selections.append(rhs_selection)
                    result.append(seq_instance)

        return result


