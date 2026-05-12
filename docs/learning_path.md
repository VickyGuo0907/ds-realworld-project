# Learning Path

This project is designed to teach the complete ML pipeline. Choose your path:

## Path 1: Notebook First (Beginner)

**Goal**: Understand the pipeline through example code

1. **Start**: Run `notebooks/1_developer_burnout_pipeline.ipynb`
   - Read markdown explanations
   - Execute code cells
   - Understand the flow
   - Time: 30-45 minutes

2. **Next**: Read `docs/guides/01_mlflow_basics.md`
   - Understand MLflow concepts
   - Learn why we track experiments
   - Time: 15-20 minutes

3. **Explore**: Look at the modules in `src/`
   - Read the docstrings
   - Find the code that the notebook used
   - Connect concepts to implementation
   - Time: 30 minutes

4. **Deep Dive**: Read `docs/guides/02_experiment_tracking.md`
   - Understand advanced tracking
   - Learn about model versioning
   - Time: 15-20 minutes

## Path 2: Code First (Intermediate)

**Goal**: Understand through production code

1. **Architecture**: Read `docs/ARCHITECTURE.md`
   - System overview
   - Module responsibilities
   - Data flow
   - Time: 20 minutes

2. **Modules**: Study the code in `src/`
   - Start with `src/data/`
   - Read docstrings and tests
   - Time: 45 minutes

3. **Run Notebook**: Execute `notebooks/1_developer_burnout_pipeline.ipynb`
   - See modules in action
   - Verify your understanding
   - Time: 30 minutes

4. **API**: Study `api/main.py`
   - Understand FastAPI basics
   - See how models are served
   - Time: 15 minutes

## Path 3: Test Driven (Advanced)

**Goal**: Learn from tests

1. **Read Tests**: Open `tests/test_data_loader.py`
   - Understand what each module does
   - Learn testing patterns
   - Time: 15 minutes

2. **Read Implementation**: Open `src/data/loader.py`
   - See how tests drive implementation
   - Understand the TDD process
   - Time: 15 minutes

3. **Continue Module by Module**:
   - Preprocessor (tests → implementation)
   - Trainer (tests → implementation)
   - MLflow integration (tests → implementation)
   - Time: 60 minutes

4. **Run Full Tests**:
   ```bash
   pytest tests/ -v --cov=src
   ```
   - Verify understanding
   - Time: 10 minutes

## Key Learning Outcomes

By completing this project, you'll understand:

### Data Science
- ✅ End-to-end ML pipeline
- ✅ Data preprocessing and feature engineering
- ✅ Multiple algorithms and when to use them
- ✅ Model evaluation and comparison
- ✅ Cross-validation for robustness

### MLOps & Production
- ✅ Experiment tracking with MLflow
- ✅ Model versioning and governance
- ✅ REST API for model serving
- ✅ Monitoring and drift detection
- ✅ Production code standards

### Software Engineering
- ✅ Modular, reusable code
- ✅ Test-driven development
- ✅ Comprehensive documentation
- ✅ Code quality and standards
- ✅ Git workflows

## Recommended Timeline

- **Quick Overview**: 1.5 hours (Path 1, steps 1-2)
- **Full Learning**: 4-5 hours (all paths, all steps)
- **Deep Mastery**: 8-10 hours (extend with second dataset)

## What's Next?

After completing this project:

1. **Apply to New Dataset**: Use the same structure for Tennessee Mental Health dataset
2. **Add Complexity**: 
   - Implement neural network model
   - Add hyperparameter tuning with Optuna
   - Create monitoring dashboard
3. **Deploy**: 
   - Run FastAPI service locally
   - Add Docker containerization
   - Deploy to cloud

## FAQ

**Q: Do I need to understand all the math?**
A: No. Focus on when to use each algorithm and how to evaluate them.

**Q: Should I memorize the code?**
A: No. Understand the structure and patterns, not the specifics.

**Q: How do I practice?**
A: Use the same code on a new dataset (Tennessee Mental Health).

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)
- [Notebook Best Practices](https://github.com/jupyter/notebook)
