from pathlib import Path
from logger_setup import logger

def get_temperature_from_log(file_path):
    txt=f'parsing temperature from {file_path.name}: '
    msg=txt+'not found'
    temp=None
    with open(file_path, 'r') as f:
        # Read the first 1000 lines of the log file
        lines = [f.readline() for _ in range(1000)]

    # Search for the specific line
    for line in lines:
        if line.startswith("fix") and "nvt temp" in line:
            # Extract the temperature values (assuming they are space-separated)
            parts = line.split()
            # Extract the target temperature value, which is the second "350" in this case
            target_temperature = parts[5]  # This is where '350' is located in your example
            if not '$' in target_temperature:
                temp=float(target_temperature)
                msg=txt+f'{temp:.2f}'
                break

    logger.info(msg)
    return temp

def get_timestep_from_log(file_path):
    txt=f'parsing timestep from {file_path.name}: '
    msg=txt+'not found'
    dt=None
    units='nd'
    units_to_time={'nd':'','metal':'ps'}
    with open(file_path, 'r') as f:
        # Read the first 1000 lines of the log file
        lines = [f.readline() for _ in range(1000)]

    # Search for the specific line
    for line in lines:
        if line.lower().startswith("units"):
            parts = line.split()
            units = parts[-1]
        if line.lower().startswith("timestep"):
            parts = line.split()
            dt = float(parts[-1])
            msg = txt + f"{dt:.6f} {units_to_time[units]}" 
    logger.info(msg)
    return dt

def parse_state_wfile(file_path):
    txt=f'parsing STATE_WFILE from {file_path.name}: '
    msg=txt+'not found'
    value=None
    with open(file_path, 'r') as f:
        # Read all lines in the plumed.input file
        lines = f.readlines()

    # Search for the line containing "STATE_WFILE"
    for line in lines:
        # Strip spaces and check for the keyword
        line = line.strip()
        if "STATE_WFILE=" in line.upper():
            # Split the line by '=' and get the second part (the value)
            value = line.split('=')[1].strip()
            value = value.split('#')[0]
            value=value.replace(' ','')
            msg=txt+value
    logger.info(msg)
    return value


def parse_bias_factor(file_path):
    txt=f'parsing bias_factor from {file_path.name}: '
    msg=txt+'not found'
    value=None
    with open(file_path, 'r') as f:
        lines = [f.readline() for _ in range(1000)]

    # Search for the line containing "STATE_WFILE"
    for line in lines:
        # Strip spaces and check for the keyword
        line = line.strip()
        if "biasfactor" in line.lower():
            value = line.split()[-1].strip()
            value=float(value)
            msg=txt+f'{value:.2f}'
    logger.info(msg)
    return value


def parse_colvar_header(file_path):
    txt=f'parsing {file_path.name} file header: '
    msg=txt+'not found'
    value=None
    with open(file_path, 'r') as f:
        headline = f.readline() 
        
    fields=headline.split()[2:]
    result={}
    ff=''
    for idx in range(len(fields)):
        result.update({idx:fields[idx]})
        ff = ff+ fields[idx] + " " 
        
        
    logger.info(txt+ff)
    return result

def parse_to_dict(input_string):
    # Split the string by spaces and ignore comments starting with '#'
    input_string = input_string.split('#')[0]  # Remove everything after '#'
    items = input_string.strip().split()  # Split by spaces and remove leading/trailing spaces

    # Create a dictionary from the key-value pairs
    result_dict = {}
    for item in items:
        key, value = item.split('=')
        result_dict[key] = value

    return result_dict


def parse_cv_wall(file_path,cv,wall='UPPER'):
    logger.debug('parse_cv_wall')
    txt=f'parsing {wall}_WALL for cv \'{cv}\' from {file_path.name}: '
    msg=txt+'not found'
    value=None
    with open(file_path, 'r') as f:
        lines = f.readlines()

    ii=0
    for line in lines:
        line = line.strip()
        # skip commented lines
        if len(line)>0 and '#' in line.split()[0]:
            continue
        
        if f"{wall}_WALL" in line.upper():
            newline=' '.join(line.split()[1:])
            result=parse_to_dict(newline)
            if result['ARG'] == cv:
                value=float(result['AT'])
                msg=txt+f'{value:.2f}'
        if ii > 200:
            break
        ii+=1

    logger.info(msg)

    return value

