# Host Engineer Memory Index

- [omni 缺 merge_ini 致 tbe/config ini 不存在](omni-missing-merge-ini-tbe-config.md) — omni add_ops_info_target 只生成 autogen JSON 从不 merge 三层 ini 到 tbe/config; binary.json 通路的 gen_opcinfo_for_socversion.sh / gen_opinfo_json_from_ini.sh 硬编码读 tbe/config/aic-<soc>-ops-info.ini; 根因=omni 无 merge_ini_files (nn gen_ops_info.cmake:239-253 有); 修=gen_binary_from_json.cmake 加 merge_ini_files_for_binary 函数对齐 nn; merge_ini_files.py 之前已搬入 omni binary_script
