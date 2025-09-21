#!/bin/bash
# scripts/profile_performance.sh

echo "⚡ Performance profiling..."

# Create output directory
mkdir -p reports/profiling

# Memory profiling
echo "🧠 Memory profiling..."
if [ -f "fusion/core/simulation.py" ]; then
    python -m memory_profiler fusion/core/simulation.py > reports/profiling/memory_profile.txt
else
    echo "Warning: fusion/core/simulation.py not found, skipping memory profiling"
fi

# Runtime profiling with py-spy (requires target process)
echo "🔍 Runtime profiling setup..."
echo "To profile a running simulation, use:"
echo "  py-spy record -o reports/profiling/runtime_profile.svg -- python -m fusion.cli.main [args]"

# Generate basic profile with cProfile
echo "🐍 Generating basic performance profile..."
if [ -f "fusion/cli/main.py" ]; then
    python -m cProfile -o reports/profiling/profile.prof -m fusion.cli.main --help > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Profile saved to reports/profiling/profile.prof"
        echo "To view interactively: snakeviz reports/profiling/profile.prof"
    else
        echo "Warning: Could not profile fusion.cli.main, module may not exist"
    fi
else
    echo "Warning: fusion/cli/main.py not found"
fi

# Create profiling helper script
cat > reports/profiling/profile_simulation.sh << 'EOF'
#!/bin/bash
# Helper script to profile a simulation run
# Usage: ./profile_simulation.sh [simulation_args]

echo "🔥 Profiling simulation with arguments: $@"

# Memory profiling
echo "📊 Memory profiling..."
python -m memory_profiler -m fusion.cli.main "$@" > reports/profiling/sim_memory_profile.txt

# Runtime profiling
echo "⚡ Runtime profiling..."
py-spy record -o reports/profiling/sim_runtime_profile.svg -- python -m fusion.cli.main "$@"

# cProfile profiling
echo "🐍 Detailed profiling..."
python -m cProfile -o reports/profiling/sim_profile.prof -m fusion.cli.main "$@"

echo "✅ Profiling complete! Check reports/profiling/ for results"
echo "📁 Generated files:"
echo "  - sim_memory_profile.txt: Memory usage analysis"
echo "  - sim_runtime_profile.svg: Runtime profiling visualization"
echo "  - sim_profile.prof: Detailed performance profile (view with snakeviz)"
EOF

chmod +x reports/profiling/profile_simulation.sh

echo "✅ Performance profiling setup complete!"
echo "📁 Generated files in reports/profiling/:"
echo "  - memory_profile.txt: Basic memory profiling"
echo "  - profile.prof: Basic performance profile"
echo "  - profile_simulation.sh: Helper script for simulation profiling"
echo ""
echo "🚀 Usage:"
echo "  View profile: snakeviz reports/profiling/profile.prof"
echo "  Profile simulation: ./reports/profiling/profile_simulation.sh [simulation_args]"
