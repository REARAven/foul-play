import random
import os
from .team_converter import export_to_packed, export_to_dict

TEAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teams")


class TeamListIterator:
    def __init__(self, team_list_file):
        try:
            with open(os.path.join(TEAM_DIR, team_list_file), "r") as f:
                lines = f.readlines()
        except (OSError, UnicodeError):
            raise ValueError("Could not read configured team list") from None
        self.team_names = [line.strip() for line in lines]
        self.index = 0

    def get_next_team(self):
        if not self.team_names:
            raise ValueError("Team list is empty")
        team_name = self.team_names[self.index]
        self.index = (self.index + 1) % len(self.team_names)
        return team_name


def load_team(name):
    if name is None:
        return "null", "", ""

    path = os.path.join(TEAM_DIR, "{}".format(name))
    if os.path.isdir(path):
        team_file_names = list()
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.startswith("."):
                    continue
                full_path = os.path.join(root, f)
                if os.path.isfile(full_path):
                    team_file_names.append(full_path)
        if not team_file_names:
            raise ValueError("No team files found in configured directory")
        file_path = random.choice(team_file_names)

    elif os.path.isfile(path):
        file_path = path
    else:
        raise ValueError("Configured team source must be a file or directory")

    try:
        with open(file_path, "r") as f:
            team_export = f.read()
    except (OSError, UnicodeError):
        raise ValueError("Could not read configured team source") from None

    try:
        return (
            export_to_packed(team_export),
            export_to_dict(team_export),
            os.path.basename(file_path),
        )
    except Exception:
        raise ValueError("Could not parse configured team source") from None
