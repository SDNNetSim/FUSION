import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

type Theme = 'light' | 'dark' | 'system'

interface UIContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  isDark: boolean
}

const UIContext = createContext<UIContextType | undefined>(undefined)

export function UIProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window !== 'undefined') {
      return (localStorage.getItem('fusion-theme') as Theme) || 'system'
    }
    return 'system'
  })

  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    const updateDarkMode = () => {
      const dark =
        theme === 'dark' ||
        (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
      setIsDark(dark)
      document.documentElement.classList.toggle('dark', dark)
    }

    updateDarkMode()

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    mediaQuery.addEventListener('change', updateDarkMode)

    return () => mediaQuery.removeEventListener('change', updateDarkMode)
  }, [theme])

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
    localStorage.setItem('fusion-theme', newTheme)
  }

  return (
    <UIContext.Provider value={{ theme, setTheme, isDark }}>
      {children}
    </UIContext.Provider>
  )
}

export function useUI() {
  const context = useContext(UIContext)
  if (context === undefined) {
    throw new Error('useUI must be used within a UIProvider')
  }
  return context
}
