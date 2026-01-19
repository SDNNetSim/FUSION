# Reporting Module TODOs

## High Priority

### Logs Directory Organization
- **Issue**: StableBaselines3 files mixed with other logs in root directory
- **Files**: `/logs/` directory structure
- **Impact**: Hard to find specific log files, poor organization
- **Solution**: Create structured subdirectories and update RL code output paths

## Medium Priority

### Multi-format Export Support
- **Issue**: Only supports basic text/log output formats
- **Impact**: Limited reporting options for different use cases
- **Solution**: Add JSON, CSV, HTML exporters with unified interface

### Progress Reporting Enhancement
- **Issue**: No real-time progress bars for long simulations
- **Impact**: Poor user experience during long runs
- **Solution**: Integrate progress library with configurable update intervals

## Low Priority

### Report Template System
- **Issue**: Fixed report format limits customization
- **Benefits**: Users could define custom report layouts
- **Solution**: Create template engine with example templates
