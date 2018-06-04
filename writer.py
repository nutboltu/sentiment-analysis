
def write_results(file_name, results):
  with open(file_name,'wb') as file:
    for index, value in enumerate(results):
        output = 'Positive' if value == 1.0 else 'Negative'
        line = str(index) + ',' + output
        file.write(line)
        file.write('\n')