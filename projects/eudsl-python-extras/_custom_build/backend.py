# Adopted from pillow and pycapnp
import os

from setuptools.build_meta import _BuildMetaBackend

EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX = "EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX"


class _CustomBuildMetaBackend(_BuildMetaBackend):
    def _convert_config_settings_to_env_var(self):
        if (
            self.config_settings
            and EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX in self.config_settings
        ):
            os.environ[
                EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX
            ] = self.config_settings.pop(EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX)

        if (
            self.config_settings
            and f"--{EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX}" in self.config_settings
        ):
            os.environ[
                EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX
            ] = self.config_settings.pop(f"--{EUDSL_PYTHON_EXTRAS_HOST_PACKAGE_PREFIX}")

    def run_setup(self, setup_script="setup.py"):
        self._convert_config_settings_to_env_var()
        return super().run_setup(setup_script)

    def build_wheel(
        self, wheel_directory, config_settings=None, metadata_directory=None
    ):
        self.config_settings = config_settings
        self._convert_config_settings_to_env_var()
        return super().build_wheel(wheel_directory, config_settings, metadata_directory)

    def build_editable(
        self, wheel_directory, config_settings=None, metadata_directory=None
    ):
        self.config_settings = config_settings
        self._convert_config_settings_to_env_var()
        return super().build_editable(
            wheel_directory, config_settings, metadata_directory
        )

    def build_sdist(self, sdist_directory, config_settings=None):
        self.config_settings = config_settings
        self._convert_config_settings_to_env_var()
        return super().build_sdist(sdist_directory, config_settings)


_backend = _CustomBuildMetaBackend()
build_wheel = _backend.build_wheel
build_editable = _backend.build_editable
build_sdist = _backend.build_sdist
