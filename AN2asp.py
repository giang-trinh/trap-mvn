#!python3

# Translate a model from AN representation into ASP representation
#
# Usage:
#   AN2asp.py original_model.an destination_model.lp
# reads the file "original_model.an" which is parsed as an AN file, translates it into ASP format
# and writes the result to "destination_model.lp".
# If already existing, the file "destination_model.lp" is erased.
#
# Limitations:
#   - Automata names must always be in double quotes
#   - Level names must always be ordered integers without quotes, starting at 0,
#       specified in numerical order in the automata declaration lines, and the maximum
#       possible level is 9
#   - At most one declaration (of automaton or transition) per line
#   - Lines not starting with a double quote (notably: #, i[nitial_context],
#       a parenthesis or a blank character) are ignored
#   - Multi-lined OCaml-like comments in (* *) are NOT ignored (their content IS parsed)
#
# TODO:
#   - Use regular expressions everywhere
#   - Handle non-quoted automata names
#   - Handle non-numerical automata levels

from sys import argv, stdin, stdout
#from os.path import exists
import re

# Input file
an = open(argv[1], "r") if len(argv) > 1 else stdin

# Output file
asp = open(argv[2], "w") if len(argv) > 2 else stdout

# Counter of local transitions
compt_trans = 0

for line in an.readlines():
  # Empty line
  if re.match("^$", line):
    asp.write("\n")
  
  # Line must start with a "
  if re.match("\"", line):
    
    # If the line contains an antomaton declaration
    if line.find("[") != -1:
      # Automaton name (+? = non-greedy *)
      m = re.search('\"(.+?)\"', line)
      if m:
        automaton = "\""+ m.group(1) + "\""
      end = line.find(']')
      # Automaton max level
      level=line[end - 1]
      automaton = "automaton_level(" + automaton + ", 0.." + level + ").\n" 
      # Print result automaton in ASP
      asp.write(automaton)

    # If the line contains a local transition declaration
    if line.find("->") != -1:
      compt_trans = compt_trans + 1
      # Automaton name (+? = non-greedy *)
      m = re.search('\"(.+?)\"', line)
      if m:
        target = "\""+ m.group(1) + "\""
      # Target level (local condition, on this automaton)
      target_lev1 = line[line.find("->") - 2]
      transition = "condition(t" + str(compt_trans) + ", " + target + ", " + str(target_lev1) + "). "
      # Bounce level
      target_lev2 = line[line.find("->") + 3]
      transition = transition + "target(t" + str(compt_trans) + ", " + target + ", " + str(target_lev2) +"). "
      # If there are external conditions (from other automata)
      if line.find("when") != -1:
        # 1st condition after "when"
        m = re.search('when \"(.+?)\"=', line)
        if m:
          cond_name = m.group(1)
        cond_level = line[line.find("=") + 1]
        transition = transition + "condition(t" + str(compt_trans) + ", \"" + cond_name + "\", " + str(cond_level) +"). "
        # 2nd and following conditions after "when"
        end = len(line) - 1
        subline = line
        while subline.find("and ") != -1:
          begin = subline.find("and ") + 3
          subline = subline[begin:end]
          i = subline.find("=")
          cond_name = subline[2:i-1]
          cond_level = subline[i+1]
          transition = transition + "condition(t" + str(compt_trans) + ", \"" + cond_name + "\", " + str(cond_level) +"). "
      # Print result transition in ASP
      if transition[-1] == " ":
        transition = transition[:-1]
      asp.write(transition + "\n")

an.close()
asp.close()

