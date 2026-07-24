"""Lê o arquivo de configuração e controla temporizadores."""

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
    """Temporizadores com nomes que não bloqueiam o programa."""

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
