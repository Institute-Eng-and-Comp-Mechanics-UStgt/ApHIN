import os
import importlib.util


def create_paper_figures(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_figures.py"):
                script_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]

                # Dynamically load the module
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Call the main function if it exists
                if hasattr(module, "main"):
                    print(f"Calling main() in {script_path}")
                    module.main()


if __name__ == "__main__":
    base_directory = os.path.dirname(__file__)
    create_paper_figures(base_directory)
