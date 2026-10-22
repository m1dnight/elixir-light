defmodule Light do
  @moduledoc """
  Turn all leds on full brightness.
  """

  @compile {:no_warn_undefined, :gpio}

  @pins [
    # red
    21,
    # green
    22,
    # blue
    27,
    # white
    26
  ]

  def start do
    for pin <- @pins do
      :gpio.set_pin_mode(pin, :output)
      :gpio.digital_write(pin, :high)
    end

    :io.format(~c"Light: all LEDs on (pins ~p)~n", [@pins])

    park()
  end

  defp park do
    receive do
    after
      60_000 -> park()
    end
  end
end
