from PyQt5 import uic

ui_file = input("ui file name (default name - main.ui):")
if ui_file == "":
    ui_file = "main.ui"

if not ui_file.endswith(".ui"):
    print("Please input a valid ui file name.")
else:
    python_file = ui_file.replace(".ui", "_ui.py")

    try:
        uic.compileUi(ui_file, open(python_file, "w"))
        print(f"Successfully compiled {ui_file} to {python_file}")
    except Exception as e:
        print(f"Failed to compile {ui_file} to {python_file}")
        print(e)