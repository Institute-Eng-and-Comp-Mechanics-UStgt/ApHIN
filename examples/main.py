import os
import importlib.util


def run_script(script_path, function_name="main", arguments=None):
    """Dynamically load and execute a function from a script."""
    if os.path.exists(script_path):
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, function_name):
            print(f"Executing {function_name}() in {script_path}")
            (
                getattr(module, function_name)(arguments)
                if arguments
                else getattr(module, function_name)()
            )
        else:
            print(f"No {function_name}() function found in {script_path}")
    else:
        print(f"Script {script_path} does not exist.")


def main():
    base_dir = os.path.dirname(__file__)

    # # Define paths for each script
    examples = dict(
        mass_spring_damper=dict(
            data_gen="data_generation/mass_spring_damper_data_generation.py",
            main="mass_spring_damper.py",
            function_name="main",
        ),
        pendulum=dict(
            data_gen="pendulum_data_generation.py",
            main="pendulum.py",
            function_name="main_various_experiments",
        ),
        disc_brake_with_hole=dict(
            data_gen="-",
            main="disc_brake_with_hole.py",
            function_name="main_various_experiments",
        ),
    )

    for example, files in examples.items():
        example_dir = os.path.join(base_dir, example)

        # Run data_generation.py if it exists
        data_gen_path = os.path.join(example_dir, files["data_gen"])
        run_script(data_gen_path)

        # Run the main file of the example
        example_main_path = os.path.join(example_dir, files["main"])
        run_script(example_main_path, function_name=files["function_name"])

    # Run paper_figures.py
    paper_figures_path = os.path.join(base_dir, "paper_figures.py")
    run_script(
        paper_figures_path,
        function_name="create_paper_figures",
        arguments=base_dir,
    )


if __name__ == "__main__":
    main()
