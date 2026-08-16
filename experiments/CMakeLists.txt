add_executable(gpt2 GPT-2/gpt2.cpp)
add_executable(mnist MNIST/MNIST-Classifier.cpp)
target_link_libraries(gpt2 PRIVATE Forge)
target_link_libraries(mnist PRIVATE Forge)

target_compile_definitions(gpt2 PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/GPT-2/Loadable/"
)

if(WIN32)
    foreach (target gpt2 mnist)
        add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENBLAS_ROOT}/bin/libopenblas.dll"
                $<TARGET_FILE_DIR:gpt2>)
    endforeach ()
endif()
