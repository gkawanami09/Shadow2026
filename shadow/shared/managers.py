"""
shared/managers.py — ConfigManager (INI read/write) and Timer (named non-blocking timers).
Ported from Overengineering² Reading Dossier, Section 2 + direct source
  Original source: robot_v.3/Python/main/Managers.py
    - ConfigManager   (lines 8-31)  — ported verbatim (json-encoded lists in an .ini)
    - Timer           (lines 34-57) — same semantics, dict instead of a numpy string
      array: set_timer(name, t) arms it; get_timer(name) returns True once expired;
      an UNSET timer returns False (that is why every consumer arms its timers with
      0.05 s at startup — see line_cam.py lines 574-580 / control.py lines 1770-1776).
Shadow2026 adaptations: none (hardware-free).
"""

import configparser
import json
import time


class ConfigManager:
    def __init__(self, config_file):
        self.__config_file = config_file
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)

    def write_variable(self, section, variable, value):
        if not self.__config.has_section(section):
            self.__config.add_section(section)
        if isinstance(value, list):
            value = json.dumps(value)
        self.__config.set(section, variable, str(value))
        with open(self.__config_file, 'w') as configfile:
            self.__config.write(configfile)

    def read_variable(self, section, variable):
        if self.__config.has_option(section, variable):
            value = self.__config.get(section, variable)
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        else:
            return None


class Timer:
    """Named non-blocking timers, one instance per process.

    get_timer(name) -> True when the timer has EXPIRED (elapsed > duration).
    get_timer of a never-set name -> False (matches OE² behaviour: an armed
    timer that never existed reads as "still running", hence the .05 s
    boot-time arming pattern)."""

    def __init__(self):
        self.__timers = {}

    def remove_timer(self, name):
        self.__timers.pop(name, None)

    def set_timer(self, name, set_time):
        self.__timers[name] = (time.perf_counter(), float(set_time))

    def get_timer(self, name):
        if name not in self.__timers:
            return False
        start, duration = self.__timers[name]
        return (time.perf_counter() - start) > duration
