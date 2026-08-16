add_executable(gpt2 GPT-2/gpt2.cpp)
add_executable(mnist MNIST/MNIST-Classifier.cpp)
target_link_libraries(gpt2 PRIVATE Forge)
target_link_libraries(mnist PRIVATE Forge)

if(WIN32)
    foreach (target gpt2 mnist)
        add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENBLAS_ROOT}/bin/libopenblas.dll"
                $<TARGET_FILE_DIR:gpt2>)
    endforeach ()
endif()

install(TARGETS gpt2 mnist RUNTIME DESTINATION . COMPONENT Runtime)
install(FILES "${CMAKE_SOURCE_DIR}/README.md" "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION . COMPONENT Runtime)
if(UNIX AND OPENBLAS_LIBRARY)
    file(REAL_PATH "${OPENBLAS_LIBRARY}" OPENBLAS_LIBRARY_REAL)
    install(FILES "${OPENBLAS_LIBRARY_REAL}" DESTINATION . RENAME libopenblas.so.0 COMPONENT Runtime)
endif()
