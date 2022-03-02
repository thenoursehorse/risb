# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
#set(all_tests basic_functions ghost one_band_cubic two_band_cubic_kanamori two_band_cubic_bilayer one_band_cubic_gf one_band_semicircular_gf two_band_semicircular_gf)

#if(embedding_ed_FOUND AND kint_FOUND)
#  set(all_tests ${all_tests} ed_one_band_cubic ed_one_band_semicircular_gf)
#endif()

if(pomerol2triqs_FOUND)
  set(all_tests ${all_tests} pomerol_one_band_cubic)
endif()

if(embedding_ed_FOUND AND kint_FOUND)
  #set(all_tests ${all_tests} three_orbital_dh)
  set(all_tests ${all_tests} ed_three_orbital_dh ed_six_site_dh)
endif()

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH} ${SANITIZER_RT_PRELOAD})
endforeach()
