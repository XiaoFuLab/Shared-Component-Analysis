# Installation Instructions

1. **Install Python**

   Ensure you have Python version 3.8.18 installed on your system.

2. **Install Required Packages**

   Use the following command to install all required packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
3. **Running the Scripts**

    The scripts to execute different methods are located in the `scripts` folder. To run the proposed method, use the command:

    ```bash
    bash scripts/office_home/run_proposed.sh
Note: The code will automatically download the dataset. However, you need to specify the data directory in the `scripts/office_home/run_*.sh` files by setting the `$DATADIR` variable.