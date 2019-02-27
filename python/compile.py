# install
# from solcx import install_solc
# install_solc('v0.5.0')
# install_solc('v0.5.3')
# install_solc('v0.5.4')
# from solcx import get_installed_solc_versions, set_solc_version
# get_installed_solc_versions()
# set_solc_version(get_installed_solc_versions()[2])


from solcx import compile_files, link_code
file_path="/home/thiagonobrega/workspace/bc-playground/Contracts/"
a = compile_files([file_path+"cc2.sol"])
z=a[list(a.keys())[0]]

z['abi']
z['bin']