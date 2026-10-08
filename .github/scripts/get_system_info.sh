#!/usr/bin/env bash

# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# get_system_info.sh - Script for 'debug' printing all the system info.

function print_header {
	local name="$1"
	if [ "$CI" == "true" ]; then
		echo "::group::${name}"
	else
		echo "**********$name**********"
	fi
}

function print_end_header {
	if [ "$CI" == "true" ]; then
		echo "::endgroup::"
	else
		echo
	fi
}

function check_L0_version {
    if command -v dpkg &> /dev/null; then
        dpkg -l | grep level-zero || \
        dpkg -l | grep libze && \
        return
    fi

    echo "libze / level-zero not installed"
}

function system_info {
	print_header "system_info"
	cat /etc/os-release | grep -oP "PRETTY_NAME=\K.*"
	cat /proc/version
	print_end_header

	print_header "VGA"
	echo "hwinfo:"
	hwinfo --display || true
	echo
	echo "lshw:"
	lshw -c video || true
	echo
	echo "lspci:"
	lspci | grep -iE 'vga|display'
	print_end_header

	print_header "L0_version"
	check_L0_version
	print_end_header

	print_header "clinfo_OpenCL"
	# The driver version of OpenCL Graphics is the compute-runtime version
	clinfo || echo "clinfo and/or OpenCL not installed"
	print_end_header

	print_header "environment_vars"
	echo "PATH=$PATH"
	echo
	echo "CPATH=$CPATH"
	echo
	echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
	echo
	echo "LIBRARY_PATH=$LIBRARY_PATH"
	echo
	echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
	print_end_header

	print_header "build_system_versions"
	gcc --version 2>/dev/null || true
	echo
	clang --version 2>/dev/null || true
	echo
	make --version 2>/dev/null || true
	print_end_header

	print_header "proc_modules"
	cat /proc/modules
	print_end_header

	print_header "proc_cmdline"
	cat /proc/cmdline
	print_end_header

	print_header "CPU_info"
	lscpu
	print_end_header

	print_header "proc_meminfo"
	cat /proc/meminfo
	print_end_header

	print_header "installed_packages"
	# Instruction below may return some minor errors
	apt list --installed 2>/dev/null || true
	print_end_header
}

# Call the function above to print system info.
system_info
