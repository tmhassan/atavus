/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly DEV: boolean
  readonly PROD: boolean
  readonly MODE: string
  readonly BASE_URL: string
  readonly VITE_API_URL?: string
  readonly VITE_APP_TITLE?: string
  // Add more custom VITE_ variables here as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
