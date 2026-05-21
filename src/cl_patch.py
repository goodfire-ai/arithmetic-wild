import random 
import copy
import itertools 
from collections import defaultdict
from typing import Any, Iterator, Sequence
from datasets import Dataset
from causalab.causal.trace import CausalTrace, Mechanism

class CounterfactualDataset:
    """
    A Dataset class for managing counterfactual data.

    Counterfactuals are alternative inputs to a neural network or causal model.
    These counterfactual inputs are used for interchange interventions, where
    the original inputs are run on the neural network or causal model and then
    features or variables are fixed to take on values they would have if the
    counterfactual were provided.

    Attributes:
        id (str): Identifier for the dataset.
    """

    def __init__(self, dataset: Dataset | None = None, id: str = "null") -> None:
        """
        Initialize a CounterfactualDataset instance.

        Args:
            dataset (Dataset, optional): An existing HuggingFace dataset to use.
                                        If provided, it must contain "input" and
                                        "counterfactual_inputs" features.
                                        Defaults to None.
            id (str, optional): Identifier for the dataset. Defaults to "null".
            **kwargs: Additional keyword arguments passed to the parent Dataset class
                     when creating a new dataset (if dataset is None).

        Raises:
            AssertionError: If required features "input" or "counterfactual_inputs"
                            are missing in the provided dataset.
        """
        self.id = id

        if dataset is not None:
            # Use the provided dataset
            # Verify it has the required features
            assert "input" in dataset.features, (
                "Provided dataset missing 'input' feature"
            )
            assert "counterfactual_inputs" in dataset.features, (
                "Provided dataset missing 'counterfactual_inputs' feature"
            )

            # # check that the first example doesn't have an input with the "raw_output" key and raise an error if it does
            # if len(dataset) > 0:
            #     first_example = dataset[0]
            #     if (
            #         isinstance(first_example["input"], dict)
            #         and "raw_output" in first_example["input"]
            #     ):
            #         raise ValueError(
            #             "The 'input' feature in the provided dataset contains a 'raw_output' key. Please ensure that the dataset only contains input data without model outputs."
            #         )

            # Initialize with the provided dataset
            self.dataset = dataset
        else:
            # Create a new empty dataset with the required features
            assert "input" in self.features
            assert "counterfactual_inputs" in self.features

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any], id: str = "null"):
        """
        Create a CounterfactualDataset from a dictionary.

        Args:
            data_dict (dict): Dictionary containing "input" and "counterfactual_inputs".

        Returns:
            CounterfactualDataset: A new CounterfactualDataset instance.
        """
        dataset = Dataset.from_dict(data_dict)
        return cls(dataset=dataset, id=id)

    @classmethod
    def from_sampler(cls, size, counterfactual_sampler, filter=None, id="null"):
        """
        Generate a dataset of counterfactual examples.

        Creates a new dataset by repeatedly sampling inputs and their counterfactuals
        using the provided sampling function, optionally filtering the samples.

        Args:
            size (int): Number of examples to generate.
            counterfactual_sampler (callable): Function that returns a dictionary
                                            with keys "input" and "counterfactual_inputs".
            filter (callable, optional): Function that takes a sample and returns a boolean
                                        indicating whether to include it. Defaults to None.

        Returns:
            CounterfactualDataset: A new CounterfactualDataset containing the generated examples.
        """
        inputs = []
        counterfactuals = []
        attempts = 0
        while len(inputs) < size:
            if attempts >= (size * 100):
                print(f"Warning: Only found {len(inputs)} valid samples after {attempts} attempts")
                break 
            attempts += 1
            sample = (
                counterfactual_sampler()
            )  # sample is a dict with keys "input" and "counterfactual_inputs"
            if filter is None or filter(sample):
                inputs.append(sample["input"])
                counterfactuals.append(sample["counterfactual_inputs"])

        # Create and return a CounterfactualDataset with the generated data
        dataset = Dataset.from_dict(
            {"input": inputs, "counterfactual_inputs": counterfactuals}
        )
        return cls(dataset=dataset, id=id)

    def display_counterfactual_data(self, num_examples=1, verbose=True):
        """
        Display examples from the dataset, showing both the original inputs
        and their corresponding counterfactual inputs.

        Args:
            num_examples (int, optional): Number of examples to display. Defaults to 1.
            verbose (bool, optional): Whether to print additional information such as
                                    dataset ID and formatting. Defaults to True.

        Returns:
            dict: A dictionary containing the displayed examples for programmatic access.
        """
        if verbose:
            print(f"Dataset '{self.id}':")

        displayed_examples = {}

        for i in range(min(num_examples, len(self))):
            example = self.dataset[i]

            if verbose:
                print(f"\nExample {i + 1}:")
                print(f"Input: {example['input']}")
                print(
                    f"Counterfactual Inputs ({len(example['counterfactual_inputs'])} alternatives):"
                )

                for j, counterfactual_input in enumerate(
                    example["counterfactual_inputs"]
                ):
                    print(f"  [{j + 1}] {counterfactual_input}")

            # Store for programmatic access
            displayed_examples[i] = {
                "input": example["input"],
                "counterfactual_inputs": example["counterfactual_inputs"],
            }

        if verbose and len(self) > num_examples:
            print(f"\n... {len(self) - num_examples} more examples not shown")

        return displayed_examples

    def add_column(self, column_name, column_data):
        """
        Add a new column to the dataset.

        Args:
            column_name (str): Name of the new column.
            column_data (list): Data for the new column.

        Raises:
            ValueError: If the length of column_data does not match the number of examples in the dataset.
        """
        if len(column_data) != len(self.dataset):
            raise ValueError(
                f"Length of {column_name} must match number of examples in dataset."
            )

        # huggingface type stubs incorrectly mark new_fingerprint as required str, but it's optional
        self.dataset = self.dataset.add_column(
            column_name,
            column_data,
            new_fingerprint=None,  # pyright: ignore[reportArgumentType]
        )

    def remove_column(self, column_name):
        """
        Remove a column from the dataset.

        Args:
            column_name (str): Name of the column to remove.
        """
        self.dataset = self.dataset.remove_columns(column_name)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get an example from the dataset by index.

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            dict: The example at the specified index, containing "input" and
                    "counterfactual_inputs".
        """
        return self.dataset[idx]

    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.

        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterate over examples in the dataset.

        Yields:
            dict: Each example containing "input" and "counterfactual_inputs".
        """
        for i in range(len(self.dataset)):
            yield self.dataset[i]


class CausalModel:
    """
    A class to represent a causal model with variables, values, parents, and mechanisms.
    Attributes:
    -----------
    variables : list
        A list of variables in the causal model.
    values : dict
        A dictionary mapping each variable to its possible values.
    parents : dict
        A dictionary mapping each variable to its parent variables.
    mechanisms : dict
        A dictionary mapping each variable to its causal mechanism.
    print_pos : dict, optional
        A dictionary specifying positions for plotting (default is None).
    """

    def __init__(
        self, variables, values, parents, mechanisms, print_pos=None, id="null"
    ):
        """
        Initialize a CausalModel instance with the given parameters.

        Parameters:
        -----------
        variables : list
            A list of variables in the causal model.
        values : dict
            A dictionary mapping each variable to its possible values.
        parents : dict
            A dictionary mapping each variable to its parent variables.
        mechanisms : dict
            A dictionary mapping each variable to its causal mechanism.
        print_pos : dict, optional
            A dictionary specifying positions for plotting (default is None).
        """
        self.variables = variables
        self.values = values
        self.parents = parents
        self.mechanisms = mechanisms
        self.id = id
        assert "raw_input" in self.variables, (
            "Variable 'raw_input' must be present in the model variables."
        )
        assert "raw_output" in self.variables, (
            "Variable 'raw_output' must be present in the model variables."
        )

        # Create children and verify model integrity
        self.children = {var: [] for var in variables}
        for variable in variables:
            assert variable in self.parents
            for parent in self.parents[variable]:
                self.children[parent].append(variable)

        # Find inputs and outputs
        self.inputs = [var for var in self.variables if len(parents[var]) == 0]
        self.outputs = copy.deepcopy(variables)
        for child in variables:
            for parent in parents[child]:
                if parent in self.outputs:
                    self.outputs.remove(parent)

        # Generate timesteps
        self.timesteps = {input_var: 0 for input_var in self.inputs}
        step = 1
        change = True
        while change:
            change = False
            copytimesteps = copy.deepcopy(self.timesteps)
            for parent in self.timesteps:
                if self.timesteps[parent] == step - 1:
                    for child in self.children[parent]:
                        copytimesteps[child] = step
                        change = True
            self.timesteps = copytimesteps
            step += 1
        self.end_time = step - 2
        for output in self.outputs:
            self.timesteps[output] = self.end_time

        # Verify that the model is valid
        for variable in self.variables:
            try:
                assert variable in self.values
            except AssertionError:
                raise ValueError(f"Variable {variable} not in values")
            try:
                assert variable in self.children
            except AssertionError:
                raise ValueError(f"Variable {variable} not in children")
            try:
                assert variable in self.mechanisms
            except AssertionError:
                raise ValueError(f"Variable {variable} not in mechanisms")
            try:
                assert variable in self.timesteps
            except AssertionError:
                raise ValueError(f"Variable {variable} not in timesteps")

            for variable2 in copy.copy(self.variables):
                if variable2 in self.parents[variable]:
                    try:
                        assert variable in self.children[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in children of {variable2}"
                        )
                    try:
                        assert self.timesteps[variable2] < self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has a later timestep than {variable}"
                        )
                if variable2 in self.children[variable]:
                    try:
                        assert variable in parents[variable2]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable} not in parents of {variable2}"
                        )
                    try:
                        assert self.timesteps[variable2] > self.timesteps[variable]
                    except AssertionError:
                        raise ValueError(
                            f"Variable {variable2} has an earlier timestep than {variable}"
                        )

        # Sort variables by timestep
        self.variables.sort(key=lambda x: self.timesteps[x])

        # Set positions for plotting
        self.print_pos = print_pos
        width = {_: 0 for _ in range(len(self.variables))}
        if self.print_pos is None:
            self.print_pos = dict()
        if "raw_input" not in self.print_pos:
            self.print_pos["raw_input"] = (0, -2)
        for var in self.variables:
            if var not in self.print_pos:
                self.print_pos[var] = (width[self.timesteps[var]], self.timesteps[var])
                width[self.timesteps[var]] += 1

        # Initializing the equivalence classes of children values
        # that produce a given parent value is expensive
        self.equiv_classes = {}

    # FUNCTIONS FOR RUNNING THE MODEL

    def run_forward(self, intervention=None):
        """
        Run the causal model forward with optional interventions.

        Parameters:
        -----------
        intervention : dict, optional
            A dictionary mapping variables to their intervened values (default is None).

        Returns:
        --------
        dict
            A dictionary mapping each variable to its computed value.
        """
        # copy intervention to avoid modifying the original dictionary
        local_intervention = copy.deepcopy(intervention)

        if local_intervention is None:
            local_intervention = {}

        # clear raw_input because it should always be recomputed
        if "raw_input" in local_intervention:
            del local_intervention["raw_input"]

        total_setting = defaultdict(None)
        length = len(list(total_setting.keys()))
        while length != len(self.variables):
            for variable in self.variables:
                for variable2 in self.parents[variable]:
                    if variable2 not in total_setting:
                        continue
                if variable in local_intervention:
                    total_setting[variable] = local_intervention[variable]
                else:
                    total_setting[variable] = self.mechanisms[variable](
                        *[total_setting[parent] for parent in self.parents[variable]]
                    )
            length = len(list(total_setting.keys()))
        return total_setting

    def run_interchange(self, input_setting, counterfactual_inputs):
        """
        Run the model with interchange interventions.

        Parameters:
        -----------
        input_setting : dict
            A dictionary mapping input variables to their values.
        counterfactual_inputs : dict
            A dictionary mapping variables to their counterfactual input settings.
            Variable names can use the format "original_var<-counterfactual_var" to specify
            different variable names in the original and counterfactual inputs.

        Returns:
        --------
        dict
            A dictionary mapping each variable to its computed value after
            interchange interventions.
        """
        interchange_intervention = copy.deepcopy(input_setting)
        for var in counterfactual_inputs:
            # Check if var contains "<-" syntax
            if "<-" in var:
                original_var, counterfactual_var = var.split("<-")
                original_var = original_var.strip()
                counterfactual_var = counterfactual_var.strip()

                # Create separate original and counterfactual dicts
                counterfactual_dict = copy.deepcopy(counterfactual_inputs[var])

                # Run forward on counterfactual to get the counterfactual variable value
                counterfactual_setting = self.run_forward(counterfactual_dict)

                # Intervene on the original variable with the counterfactual variable's value
                interchange_intervention[original_var] = counterfactual_setting[
                    counterfactual_var
                ]
            else:
                # Original behavior: both original and counterfactual use the same variable name
                setting = self.run_forward(counterfactual_inputs[var])
                interchange_intervention[var] = setting[var]
        return self.run_forward(interchange_intervention)

    def new_raw_input(self, input_dict):
        """
        Compute and set the raw_input field for an input dictionary.

        This is a convenience method that runs the causal model forward on the input
        and updates the input dictionary with the computed raw_input value.

        Parameters:
        -----------
        input_dict : dict
            A dictionary mapping input variables to their values. This dictionary
            will be modified in-place to include the "raw_input" field.

        Returns:
        --------
        None
            The input_dict is modified in-place.
        """
        # Remove raw_input if it exists to ensure fresh computation
        if "raw_input" in input_dict:
            del input_dict["raw_input"]
        result = self.run_forward(input_dict)
        input_dict["raw_input"] = result["raw_input"]

    # FUNCTIONS FOR SAMPLING INPUTS AND GENERATING DATASETS

    def sample_intervention(self, filter_func=None):
        """
        Sample a random intervention that satisfies an optional filter.

        Parameters:
        -----------
        filter_func : function, optional
            A function that takes an intervention and returns a boolean indicating
            whether it satisfies the filter (default is None).

        Returns:
        --------
        dict
            A dictionary mapping variables to their sampled intervention values.
        """
        filter_func = (
            filter_func if filter_func is not None else lambda x: len(x.keys()) > 0
        )
        intervention = {}
        while not filter_func(intervention):
            intervention = {}
            while len(intervention.keys()) == 0:
                for var in self.variables:
                    if var in self.inputs or var in self.outputs:
                        continue
                    if random.choice([0, 1]) == 0:
                        intervention[var] = random.choice(self.values[var])
        return intervention

    def sample_input(self, filter_func=None):
        """
        Sample a random input that satisfies an optional filter when run through the model.

        Parameters:
        -----------
        filter_func : function, optional
            A function that takes a setting and returns a boolean indicating
            whether it satisfies the filter (default is None).

        Returns:
        --------
        dict
            A dictionary mapping input variables to their sampled values.
        """
        filter_func = filter_func if filter_func is not None else lambda x: True
        input_setting = {var: random.choice(self.values[var]) for var in self.inputs}
        total = self.run_forward(intervention=input_setting)

        while not filter_func(total):
            input_setting = {
                var: random.choice(self.values[var]) for var in self.inputs
            }
            total = self.run_forward(intervention=input_setting)

        input_setting["raw_input"] = total["raw_input"]
        return input_setting

    def generate_dataset(self, size, input_sampler=None, filter_func=None):
        """
        Generate a dataset of inputs.

        Parameters:
        -----------
        size : int
            The number of samples to generate.
        input_sampler : function, optional
            A function that samples inputs (default is self.sample_input).
        filter_func : function, optional
            A function that takes an input and returns a boolean indicating
            whether it satisfies the filter (default is None).

        Returns:
        --------
        Dataset
            A Hugging Face Dataset with an "input" field.
        """
        if input_sampler is None:
            input_sampler = self.sample_input
        inputs = []
        while len(inputs) < size:
            inp = input_sampler()
            if filter_func is None or filter_func(inp):
                inputs.append(inp)
        # Create and return a Hugging Face Dataset with a single "input" field.
        return Dataset.from_dict({"input": inputs})

    def label_counterfactual_data(self, examples, target_variables: Sequence[str]):
        """
        Labels examples with results from running interchange interventions.

        Accepts the public causalab list-of-CounterfactualExample format (where
        "input" / "counterfactual_inputs" are CausalTrace objects) and bridges
        to the legacy dict-based run_interchange.

        Returns the examples with "label" and "setting" fields added.
        """
        def _to_input_dict(t):
            # Restrict to input variables only — downstream variables must be
            # recomputed by run_forward, otherwise stale values like raw_output
            # leak through and the interchange becomes a no-op.
            full = dict(t._values) if hasattr(t, "_values") else dict(t)
            return {var: full[var] for var in self.inputs if var in full}

        labels = []
        settings = []

        for example in examples:
            input_dict = _to_input_dict(example["input"])
            counterfactual_dicts = [_to_input_dict(t) for t in example["counterfactual_inputs"]]

            if len(counterfactual_dicts) == 1 and len(target_variables) > 1:
                counterfactual_dicts = counterfactual_dicts * len(target_variables)

            assert len(target_variables) <= len(counterfactual_dicts), (
                f"target_variables has {len(target_variables)} elements but counterfactual_inputs only has {len(counterfactual_dicts)}"
            )

            counterfactual_dict = {}
            for i, var_element in enumerate(target_variables):
                if isinstance(var_element, list):
                    for var in var_element:
                        counterfactual_dict[var] = counterfactual_dicts[i]
                else:
                    counterfactual_dict[var_element] = counterfactual_dicts[i]

            setting = self.run_interchange(input_dict, counterfactual_dict)
            labels.append(setting["raw_output"])
            settings.append(dict(setting))

        result = []
        for i, example in enumerate(examples):
            result.append({**example, "label": labels[i], "setting": settings[i]})
        return result

    def label_data_with_variables(self, dataset, target_variables):
        """
        Labels a dataset based on variable settings from running the forward model.

        Takes a dataset of inputs, runs the forward model on each input, and assigns
        a unique label ID based on the values of the specified target variables.

        Parameters:
        -----------
        dataset : Dataset
            Dataset containing "input" field.
        target_variables : list
            List of variable names to use for labeling.

        Returns:
        --------
        tuple
            A tuple containing:
                - Dataset: A new dataset with inputs and corresponding labels.
                - dict: A mapping from concatenated target variable values to label IDs.
        """
        inputs = []
        labels = []
        label_to_setting = {}

        new_id = 0
        for example in dataset:
            # Store input
            inputs.append(example["input"])

            # Run forward model and get target variable values
            setting = self.run_forward(example["input"])
            target_labels = [str(setting[var]) for var in target_variables]

            # Assign or create a label ID
            label_key = "".join(target_labels)
            if label_key in label_to_setting:
                id_value = label_to_setting[label_key]
            else:
                id_value = new_id
                label_to_setting[label_key] = new_id
                new_id += 1

            labels.append(id_value)

        return Dataset.from_dict({"input": inputs, "label": labels}), label_to_setting

    def can_distinguish_with_dataset(
        self, dataset, target_variables1, target_variables2
    ):
        """
        Check if the model can distinguish between two sets of target variables
        using interchange interventions on a counterfactual dataset.
        """

        count = 0
        for example in dataset:
            input = example["input"]
            counterfactual_inputs = example["counterfactual_inputs"]
            assert len(counterfactual_inputs) == 1

            setting1 = self.run_interchange(
                input, {var: counterfactual_inputs[0] for var in target_variables1}
            )
            if target_variables2 is not None:
                setting2 = self.run_interchange(
                    input, {var: counterfactual_inputs[0] for var in target_variables2}
                )
                if setting1["raw_output"] != setting2["raw_output"]:
                    count += 1
            else:
                if setting1["raw_output"] != self.run_forward(input)["raw_output"]:
                    count += 1

        proportion = count / len(dataset)

        # logger.debug(
        #     f"Can distinguish between {target_variables1} and {target_variables2}: {count} out of {len(dataset)} examples"
        # )
        # logger.debug(f"Proportion of distinguishable examples: {proportion:.2f}")

        return {"proportion": proportion, "count": count}

    def generate_equiv_classes(self):
        """
        Generate equivalence classes for each variable.

        This method computes, for each non-input variable, the sets of parent values
        that produce each possible value of the variable.
        """
        for var in self.variables:
            if var in self.inputs or var in self.equiv_classes:
                continue
            self.equiv_classes[var] = {val: [] for val in self.values[var]}
            for parent_values in itertools.product(
                *[self.values[par] for par in self.parents[var]]
            ):
                value = self.mechanisms[var](*parent_values)
                self.equiv_classes[var][value].append(
                    {par: parent_values[i] for i, par in enumerate(self.parents[var])}
                )

    def find_live_paths(self, intervention):
        """
        Find all live causal paths in the model given an intervention.

        A live path is a sequence of variables where changing the value of one
        variable can affect the value of the next variable in the sequence.

        Parameters:
        -----------
        intervention : dict
            A dictionary mapping variables to their intervened values.

        Returns:
        --------
        dict
            A dictionary mapping path lengths to lists of paths.
        """
        actual_setting = self.run_forward(intervention)
        paths = {1: [[variable] for variable in self.variables]}
        step = 2
        while True:
            paths[step] = []
            for path in paths[step - 1]:
                for child in self.children[path[-1]]:
                    actual_cause = False
                    for value in self.values[path[-1]]:
                        newintervention = copy.deepcopy(intervention)
                        newintervention[path[-1]] = value
                        counterfactual_setting = self.run_forward(newintervention)
                        if counterfactual_setting[child] != actual_setting[child]:
                            actual_cause = True
                    if actual_cause:
                        paths[step].append(copy.deepcopy(path) + [child])
            if len(paths[step]) == 0:
                break
            step += 1
        del paths[1]
        return paths

    def sample_input_tree_balanced(self, output_var=None, output_var_value=None):
        """
        Sample an input that leads to a specific output value using a balanced tree approach.

        Parameters:
        -----------
        output_var : str, optional
            The output variable to target (default is the first output variable).
        output_var_value : any, optional
            The desired value for the output variable (default is a random choice).

        Returns:
        --------
        dict
            A dictionary mapping input variables to their sampled values.
        """
        assert output_var is not None or len(self.outputs) == 1
        self.generate_equiv_classes()

        if output_var is None:
            output_var = self.outputs[0]
        if output_var_value is None:
            output_var_value = random.choice(self.values[output_var])

        def create_input(var, value, input_dict={}):
            """
            Recursively create an input that leads to the specified value for a variable.

            Parameters:
            -----------
            var : str
                The variable to target.
            value : any
                The desired value for the variable.
            input_dict : dict, optional
                The input dictionary to build upon (default is an empty dictionary).

            Returns:
            --------
            dict
                The updated input dictionary.
            """
            parent_values = random.choice(self.equiv_classes[var][value])
            for parent in parent_values:
                if parent in self.inputs:
                    input_dict[parent] = parent_values[parent]
                else:
                    create_input(parent, parent_values[parent], input_dict)
            return input_dict

        input_setting = create_input(output_var, output_var_value)
        for input_var in self.inputs:
            if input_var not in input_setting:
                input_setting[input_var] = random.choice(self.values[input_var])
        return input_setting

    def get_path_maxlen_filter(self, lengths):
        """
        Get a filter function that checks if the maximum length of any live path
        is in a given set of lengths.

        Parameters:
        -----------
        lengths : list or set
            A list or set of path lengths to check against.

        Returns:
        --------
        function
            A filter function that takes a setting and returns a boolean.
        """

        def check_path(total_setting):
            """
            Check if the maximum length of any live path is in the specified lengths.

            Parameters:
            -----------
            total_setting : dict
                A dictionary mapping variables to their values.

            Returns:
            --------
            bool
                True if the maximum length is in the specified lengths, False otherwise.
            """
            input_setting = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input_setting)
            max_len = max(
                [length for length in paths.keys() if len(paths[length]) != 0]
            )
            if max_len in lengths:
                return True
            return False

        return check_path

    def get_partial_filter(self, partial_setting):
        """
        Get a filter function that checks if a setting matches a partial setting.

        Parameters:
        -----------
        partial_setting : dict
            A dictionary mapping variables to their desired values.

        Returns:
        --------
        function
            A filter function that takes a setting and returns a boolean.
        """

        def compare(total_setting):
            """
            Check if a setting matches the partial setting.

            Parameters:
            -----------
            total_setting : dict
                A dictionary mapping variables to their values.

            Returns:
            --------
            bool
                True if the setting matches the partial setting, False otherwise.
            """
            for var in partial_setting:
                if total_setting[var] != partial_setting[var]:
                    return False
            return True

        return compare

    def get_specific_path_filter(self, start, end):
        """
        Get a filter function that checks if there is a live path from a start
        variable to an end variable.

        Parameters:
        -----------
        start : str
            The start variable of the path.
        end : str
            The end variable of the path.

        Returns:
        --------
        function
            A filter function that takes a setting and returns a boolean.
        """

        def check_path(total_setting):
            """
            Check if there is a live path from the start variable to the end variable.

            Parameters:
            -----------
            total_setting : dict
                A dictionary mapping variables to their values.

            Returns:
            --------
            bool
                True if there is a live path from start to end, False otherwise.
            """
            input_setting = {var: total_setting[var] for var in self.inputs}
            paths = self.find_live_paths(input_setting)
            for k in paths:
                for path in paths[k]:
                    if path[0] == start and path[-1] == end:
                        return True
            return False

        return check_path

    def new_trace(self, inputs: dict[str, Any] | None = None):
        """
        Create a new trace for running this causal model.

        Parameters:
        -----------
        inputs : dict, optional
            Input variables to set (default is None).
            Should only contain input variables - computed variables will be
            automatically computed from inputs.

        Returns:
        --------
        CausalTrace
            A new trace object for setting inputs/interventions and getting values.
        """
        # Bridge legacy `dict[str, function]` mechanisms (positional parent args)
        # into the public `Mechanism` dataclass that `CausalTrace` expects.
        wrapped: dict[str, Mechanism] = {}
        for var, fn in self.mechanisms.items():
            parents = list(self.parents[var])
            wrapped[var] = Mechanism(
                parents=parents,
                compute=(lambda t, fn=fn, parents=parents: fn(*[t[p] for p in parents])),
            )
        return CausalTrace(mechanisms=wrapped, inputs=inputs)