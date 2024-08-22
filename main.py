# Multiple testcase options:

# 1) Run test case with only the blood flow model
# import testcases.testcase_blood_flow_model

# 2) Run test case with the blood flow considering the phase separation of red blood cells (RBCs) (iterative model)
# import testcases.testcase_blood_flow_iterative_model

# 3) Run test case with the blood flow model considering vascular distensibility
# import testcases.testcase_distensibility

# 4) Run test case to tune diameters or transmissibilitites with the inverse model
# import testcases.testcase_inverse_problem

# 5) Run test case to tune boundary pressures with the inverse model
# import testcases.testcase_bc_tuning

# Import a testcase based on the options above. 
# If not stated otherwise, testcases.testcase_blood_flow_model is executed by default
if __name__ == "__main__":
    import testcases.testcase_particle_tracking # testcases.testcase_blood_flow_model 
