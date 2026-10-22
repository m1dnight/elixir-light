defmodule Light.MixProject do
  use Mix.Project

  def project do
    [
      app: :light,
      version: "0.1.0",
      elixir: "~> 1.13",
      deps: deps(),
      # This board has flash encryption burned into its eFuses, so the plaintext
      # writes of mix atomvm.esp32.flash would not boot. `mix flash` lets
      # esptool encrypt on the way in instead.
      aliases: [flash: ["atomvm.packbeam", &flash/1]],
      atomvm: [
        start: Light,
        chip: "esp32"
      ]
    ]
  end

  @port "/dev/cu.usbserial-1122301"
  @main_avm_offset "0x250000"

  defp flash(_args) do
    port = System.get_env("PORT") || @port
    cmd = "esptool --port #{port} write-flash --encrypt #{@main_avm_offset} light.avm"
    Mix.shell().info(cmd)
    case Mix.shell().cmd(cmd) do
      0 -> :ok
      status -> Mix.raise("esptool exited with status #{status}")
    end
  end

  def application do
    [extra_applications: []]
  end

  defp deps do
    [
      {:atomvm, "~> 0.7.0-beta.0", runtime: false},
      {:exatomvm,
       github: "atomvm/ExAtomVM", ref: "a99323a1054adda1ec00e9997bc41bc7317a3826", runtime: false},
      # ExAtomVM runs esptool inside this embedded Python when flashing.
      {:pythonx, "~> 0.4.0", runtime: false},
      # Used by mix atomvm.esp32.install to download the VM image.
      {:req, "~> 0.7.0", runtime: false}
    ]
  end
end
